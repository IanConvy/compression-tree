import numpy as np
import tensorflow as tf
import compress as compr
import data
from tqdm import tqdm

class Trainer():
    def __init__(self, num_pixels, tree_path):
        self.compr_tree = compr.Compression_Tree(num_pixels)
        self.compr_tree.load(tree_path)
        self.create_trainer_ops()

    def create_trainer_ops(self):
        self.weights = tf.get_variable('Weights', initializer = np.random.normal(scale = 0.1,  size = [10, self.compr_tree.output_size]).astype(np.float32))
        self.label_batch = tf.placeholder(tf.float32)
        compr_output = tf.einsum('ai,aj->aij', self.compr_tree.left_output, self.compr_tree.right_output)
        compr_images = tf.reshape(compr_output, [tf.shape(compr_output)[0], -1])
        self.predictions = tf.einsum('ij,aj->ai', self.weights, compr_images)
        compare = tf.equal(tf.argmax(self.predictions, axis = 1), tf.argmax(self.label_batch, axis = 1))
        sqr_diff = (self.label_batch - self.predictions) ** 2
        self.loss = 0.5 * tf.reduce_mean(sqr_diff)
        self.optimize = tf.train.RMSPropOptimizer(0.0005).minimize(self.loss, var_list = [self.weights])
        self.accuracy = tf.reduce_mean(tf.cast(compare, tf.float32))

    def train(self, train_data, test_data, epochs = 30, batch_size = 20):
        (train_images, train_labels) = (train_data[0].astype(np.float32), train_data[1].astype(np.int32))
        (test_images, test_labels) = (test_data[0].astype(np.float32), test_data[1].astype(np.int32))
        total_batches = train_images.shape[0] / batch_size
        one_hot_train = data.one_hot(train_labels)
        one_hot_test = data.one_hot(test_labels)
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            for epoch in range(epochs):
                batch_itr = data.batch_generator(train_images, one_hot_train, batch_size)
                for (image_batch, label_batch) in tqdm(batch_itr, total = total_batches):
                    sess.run(self.optimize, feed_dict = {self.compr_tree.images:image_batch, self.label_batch:label_batch})
                accuracy = sess.run(self.accuracy, feed_dict = {self.compr_tree.images:test_images, self.label_batch:one_hot_test})
                print('Epoch {}: {}'.format(epoch, accuracy))

train_path = 'data/8_train'
test_path = 'data/8_test'
tree_path = 'models/64_005'

trainer = Trainer(64, tree_path)
train_data = data.load_data(train_path)
test_data = data.load_data(test_path)
trainer.train(train_data, test_data)
