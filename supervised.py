import numpy as np
import tensorflow as tf
import compress as compr
import data
from tqdm import tqdm

class Trainer():
    """
    Performs supervised learning on MNIST images using data compression.

    This class builds a tensorflow graph on top of a compression tree and
    optimizes a weight matrix to do image classification. It is used primarily
    as a sanity check for the output of the compression tree, and is not intended
    to achieve the best possible accuracy.

    Attributes:
        compr_tree (object): An instance of Compression_Tree to use for compression.
        weights (tensor): The parameters to be optimized for image classification.
        label_batch (tensor): Labels to be used while training on mini-batch.
        optimize (tensor): Run for an iteration of RMSProp
        accuracy (tensor): Run to get the accuracy of the model.
    """

    def __init__(self, num_pixels, tree_path):
        """
        Creates instance of Compression_Tree and additional tensor operations.

        Parameters:
            num_pixels (integer): Number of pixels in the images.
            tree_path (string): Directory containing the compression tree arrays.
        """
        self.compr_tree = compr.Compression_Tree(num_pixels)
        self.compr_tree.load(tree_path)
        self.create_trainer_ops()

    def create_trainer_ops(self):
        """
        Creates all tensor operations necessary for the supervised learning.

        This function generates the tensorflow graph that connects to the output
        tensors of the compression tree and classifies the compressed images.
        It initializes the weight variable using a random normal distribution, with
        the number of rows equal to the number of different digits and the number
        of columns equal to the output size of the compression tree. The images
        are constructed as the kronecker product of the left and right outputs from
        the tree.

        To optimize the weight parameters, the MSE is calculated for the image batch
        using the training labels and is then minimized using TensorFlow's pre-built
        RMSProp algorithm (other optimizers could also be used). The accuracy can
        be easily cacluated by viewing the product of the weights and image as a
        one-hot vector whose largest value encodes the predicted digit. This is then
        compared to the encoded value in the actual one-hot label.
        """
        #Create weight variable and the label placeholder.
        weight_initializer = np.random.normal(scale = 0.1,  size = [10, self.compr_tree.output_size])
        self.weights = tf.get_variable('Weights', initializer = weight_initializer.astype(np.float32))
        self.label_batch = tf.placeholder(tf.float32)

        #Get the outputs of the compression tree and combine to create the image.
        compr_output = tf.einsum('ai,aj->aij', self.compr_tree.left_output, self.compr_tree.right_output)
        compr_images = tf.reshape(compr_output, [tf.shape(compr_output)[0], -1])

        #Operate on the images using the weights and calculate the MSE to optimize.
        predictions = tf.einsum('ij,aj->ai', self.weights, compr_images)
        sqr_diff = (self.label_batch - predictions) ** 2
        loss = 0.5 * tf.reduce_mean(sqr_diff)
        self.optimize = tf.train.RMSPropOptimizer(0.0005).minimize(loss, var_list = [self.weights])

        #Calculate the accuracy
        compare = tf.equal(tf.argmax(predictions, axis = 1), tf.argmax(self.label_batch, axis = 1))
        self.accuracy = tf.reduce_mean(tf.cast(compare, tf.float32))

    def train(self, train_data, test_data, epochs = 30, batch_size = 20):
        """
        Do supervised training on the compressed images.

        This function takes training and test data (converting to 32 byte),
        converts the integer labels to one-hot vectors, and then runs the given
        number of epochs. For each epoch, the training data is seperated into
        mini-batches of the specified size, and then for each batch the optimizer
        is run. After all the batches have been trained on, the accuracy on the
        test data is calculated and displayed.

        Parameters:
            train_data (tuple): Tuple of training images / integer labels.
            test_data (tuple): Tuple of test images / integer labels.
            epochs (integer): Number of epochs to train.
            batch_size (integer): Size of the mini-batches.
        """
        #Prepare labels and ensure proper data type.
        train_images = train_data[0].astype(np.float32)
        train_labels = train_data[1].astype(np.int32)
        test_images = test_data[0].astype(np.float32)
        test_labels = test_data[1].astype(np.int32)
        one_hot_train = data.one_hot(train_labels)
        one_hot_test = data.one_hot(test_labels)

        #Train the model.
        total_batches = train_images.shape[0] / batch_size
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            for epoch in range(epochs):
                batch_itr = data.batch_generator(train_images, one_hot_train, batch_size)
                for (image_batch, label_batch) in tqdm(batch_itr, total = total_batches):
                    sess.run(self.optimize, feed_dict = {self.compr_tree.images:image_batch, self.label_batch:label_batch})
                accuracy = sess.run(self.accuracy, feed_dict = {self.compr_tree.images:test_images, self.label_batch:one_hot_test})
                print('Epoch {}: {:.1f}% Accuracy'.format(epoch, accuracy * 100))

if __name__ == '__main__':
    #Specify data and model paths.
    train_path = 'data/8_train'
    test_path = 'data/8_test'
    tree_path = 'models/64_005'
    num_pixels = 64

    #Load data and train.
    trainer = Trainer(num_pixels, tree_path)
    train_data = data.load_data(train_path)
    test_data = data.load_data(test_path)
    trainer.train(train_data, test_data)
