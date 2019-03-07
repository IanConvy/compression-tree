import pathlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import unsupervised as train
import data

class Compression_Tree():
    def __init__(self, num_pixels):
        self.nodes = []
        self.num_layers = int(np.log2(num_pixels)) - 1
        for layer in range(self.num_layers):
            node_layer = []
            num_nodes = int(num_pixels / (2 ** (layer + 1)))
            for position in range(num_nodes):
                new_node = Node(layer, position)
                node_layer.append(new_node)
            self.nodes.append(node_layer)

    def load(self, path):
        for layer in self.nodes:
            for node in layer:
                with tf.variable_scope(node.scope):
                    array_path = pathlib.PurePath(path).joinpath(node.scope + '.npy')
                    isom_array = np.load(str(array_path))
                    node.output_size = isom_array.shape[0]
                    node.isom = tf.constant(isom_array)
        self.output_size = self.nodes[-1][0].output_size * self.nodes[-1][1].output_size
        self.create_tree_ops()

    def train(self, images, threshold, save_path = False, size_limit =  5 * 10**8):
        tf.enable_eager_execution()
        if save_path:
            pathlib.Path(save_path).mkdir(parents = True, exist_ok = True)
        trainer = train.Compression_Trainer(self.num_layers)
        for layer in self.nodes:
            for node in layer:
                print('Layer {}, Node {} training'.format(node.layer, node.position))
                isom_array = trainer.train_node(node.layer, node.position, images, threshold, size_limit)
                node.isom = tf.constant(isom_array)
                node.output_size = isom_array.shape[0]
                if save_path:
                    isom_path = pathlib.PurePath(save_path).joinpath(node.scope)
                    np.save(str(isom_path), isom_array)
        print('Training Complete')

    def create_tree_ops(self):
        self.images = tf.placeholder(tf.float32, shape = [None, None, None])
        for layer in self.nodes:
            for node in layer:
                left_pos = 2 * node.position
                if node.layer == 0:
                    left_in = self.images[:, :, left_pos]
                    right_in = self.images[:, :, left_pos + 1]
                else:
                    left_in = self.nodes[node.layer - 1][left_pos].output
                    right_in = self.nodes[node.layer - 1][left_pos + 1].output
                node.create_compression_ops(left_in, right_in)
        self.left_output = self.nodes[-1][0].output
        self.right_output = self.nodes[-1][1].output

    def compress(self, images):
        with tf.Session() as sess:
            (left_out, right_out) = sess.run((self.left_output, self.right_output), feed_dict = {self.images:images})
        return (left_out, right_out)

    def plot_dimmensions(self):
        plot_side = int(np.ceil(np.sqrt(self.num_layers)))
        (fig, ax_array) = plt.subplots(plot_side, plot_side)
        ax_vect = ax_array.flatten()
        for (layer_num, layer) in enumerate(self.nodes):
            x_values = []
            y_values = []
            for (position, node) in enumerate(layer):
                x_values.append(position)
                y_values.append(node.output_size)
                ax_vect[layer_num].plot(x_values, y_values, marker = 'o', linestyle = 'None')
        plt.show()

class Node():
    def __init__(self, layer, position):
        self.layer = layer
        self.position = position
        self.scope = '{}_{}'.format(layer, position)

    def create_compression_ops(self, left_in, right_in):
        outer = tf.einsum('ab,ac->abc', left_in, right_in)
        batch_size = tf.shape(left_in)[0]
        kron = tf.reshape(outer, [batch_size, -1])
        self.output = tf.einsum('ij,aj->ai', self.isom, kron)
