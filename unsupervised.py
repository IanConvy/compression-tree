import pathlib
import numpy as np
import tensorflow as tf
from tqdm import tqdm

class Compression_Trainer():
    def __init__(self, num_layers):
        self.training_nodes = {}
        for layer in range(num_layers):
            self.training_nodes[layer] = {}

    def train_node(self, layer, position, train_images, threshold, size_limit):
        node = Training_Node(layer, position)
        self.training_nodes[layer][position] = node
        (left_prev_out, right_prev_out) = self.feed_forward(train_images, layer, position, 0, 1)
        density_side_len = tf.size(left_prev_out) * tf.size(right_prev_out)
        node.create_train_params(density_side_len, threshold)
        num_images = train_images.shape[0]
        batch_itr = get_batch_itr(density_side_len, size_limit, num_images)
        if not batch_itr:
            print('Size limit exceeded')
            return
        with tqdm(total = num_images) as pbar:
            for (batch_start, batch_size) in batch_itr:
                pbar.set_postfix({'Batch Size' : batch_size})
                (left_in, right_in) = self.feed_forward(train_images, layer, position, batch_start, batch_size)
                node.add_batch(left_in, right_in, batch_size)
                pbar.update(batch_size)
        node.diagonalize(num_images)
        print('Output Size = {}'.format(node.output_size))
        return node.isom_var.numpy()

    def feed_forward(self, images, layer, position, batch_start, batch_size):
        left_pos = 2 * position
        right_pos = left_pos + 1
        if layer == 0:
            left_in = tf.constant(images[batch_start : batch_start + batch_size, :, left_pos])
            right_in = tf.constant(images[batch_start : batch_start + batch_size, :, right_pos])
        else:
            (left_prev_left_in, left_prev_right_in) = self.feed_forward(images, layer - 1, left_pos, batch_start, batch_size)
            (right_prev_left_in, right_prev_right_in) = self.feed_forward(images, layer - 1, right_pos, batch_start, batch_size)
            left_prev_node = self.training_nodes[layer - 1][left_pos]
            right_prev_node = self.training_nodes[layer - 1][right_pos]
            left_in = left_prev_node.compress(left_prev_left_in, left_prev_right_in, batch_size)
            right_in = right_prev_node.compress(right_prev_left_in, right_prev_right_in, batch_size)
        return (left_in, right_in)

class Training_Node():
    def __init__(self, layer, position):
        self.scope = '{}_{}'.format(layer, position)

    def create_train_params(self, density_side_len, threshold):
        self.threshold = threshold
        with tf.variable_scope(self.scope):
            self.density_var = tf.get_variable('Batch_Density', initializer = tf.zeros([density_side_len, density_side_len]))

    def add_batch(self, left_in, right_in, batch_size):
        outer = tf.einsum('ab,ac->abc', left_in, right_in)
        kron = tf.reshape(outer, [batch_size, -1])
        density = tf.einsum('ab,ac->abc', kron, kron)
        batch_density = tf.reduce_sum(density, axis = 0)
        tf.assign_add(self.density_var, batch_density)

    def diagonalize(self, num_total_images):
        herm_matrix = self.density_var / num_total_images
        trace = tf.trace(herm_matrix)
        (eig_values, eig_vects) = tf.linalg.eigh(herm_matrix)
        cum_eigs = tf.cumsum(eig_values)
        ratios = cum_eigs / trace
        thresh_test = tf.cast(ratios <= self.threshold, tf.int32)
        trunc_index = tf.reduce_sum(thresh_test)
        trunc_isom = tf.transpose(eig_vects[:, trunc_index:])
        with tf.variable_scope(self.scope):
            self.isom_var = tf.get_variable('Isometry', initializer = trunc_isom)
        self.output_size = int(tf.shape(self.isom_var)[0])

    def compress(self, left_in, right_in, batch_size):
        outer = tf.einsum('ab,ac->abc', left_in, right_in)
        kron = tf.reshape(outer, [batch_size, -1])
        compr_out = tf.einsum('ij,aj->ai', self.isom_var, kron)
        return compr_out

def get_batch_itr(density_side_len, limit, num_images):
    matrix_size = int((density_side_len)**2)
    max_batch = min(limit // matrix_size, num_images)
    if max_batch == 0:
        return []
    clean_end = num_images // max_batch
    excess = num_images % max_batch
    batch_start = 0
    while batch_start < num_images:
        if batch_start < max_batch * clean_end:
            batch_size = max_batch
        else:
            batch_size = excess
        yield (batch_start, batch_size)
        batch_start += batch_size
