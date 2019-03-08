import pathlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import unsupervised as train
import data

class Compression_Tree():
    """
    Tensor network for unsupervised data compression.

    This class contains the unsupervised data compression model used to reduce
    the initial featurization space of the MNIST images. It can either train a
    new tree using the 'unsupervised' module or load an existing tree, but due
    to limitations in Tensorflow's current Eager Execution mode it cannot generate
    a new tree and compress images in the same python session. The tree consists
    of 'Node' objects which are each responsible for compressing two input arrays
    (from nodes in the previous layer) into a single output array that is then fed
    into the next layer. Each tree has a defined threshold value that determines
    how much the data is compressed, with larger values leading to smaller feature
    spaces.

    Attributes:
        nodes (list): Nested list with layers of nodes in order of decreasing size.
        num_layers (integer): Number of layers in tree.
        output_size (integer): Size of the compressed feature space.
        images (tensor): Placeholder for the images to be compressed.
        left_output (tensor): Output of the first half of the tree.
        right_output (tensor): Output of the second half of the tree.
    """

    def __init__(self, num_pixels):
        """
        Generates the node structure of the tree.

        This function simply determines the size of the tree from the specified
        number of pixels and then assigns nodes to the proper layer and position
        while adding them to a list. The layers are created in order of decreasing
        size, which is the same order that they will be used for compression.
        The nodes do not, at this point, contain any operations. Note that different
        sized images cannot be compressed with the same tree.

        Parameters:
            num_pixels (integer): Number of pixels to expect in the images.
        """
        self.nodes = []
        self.num_layers = int(np.log2(num_pixels)) - 1  #-1 since the tree terminates with two top nodes.
        for layer in range(self.num_layers):
            node_layer = []
            num_nodes = int(num_pixels / (2 ** (layer + 1)))  #Each layer divides the previous layer size by 2.
            for position in range(num_nodes):
                new_node = Node(layer, position)
                node_layer.append(new_node)
            self.nodes.append(node_layer)

    def load(self, path):
        """
        Create tree from a saved model.

        This function takes a set of saved numpy arrays and builds them into tensors
        which are then assigned to the appropriate node. The files are expected to
        have a naming scheme 'layer_position.npy', where 'layer' and 'position' are
        integers which specify which array belongs to which node. After the tree is
        loaded, it can be used to compress images and possibly form the base of some
        other learning model (such as in 'supervised.py').

        Parameters:
            path (string): Directory where the arrays are saved.
        """
        for layer in self.nodes:
            for node in layer:
                with tf.variable_scope(node.scope):
                    array_path = pathlib.PurePath(path).joinpath(node.scope + '.npy')
                    isom_array = np.load(str(array_path))
                    node.output_size = isom_array.shape[0]
                    node.isom = tf.constant(isom_array)
        self.output_size = self.nodes[-1][0].output_size * self.nodes[-1][1].output_size
        self.connect_tree()

    def generate(self, images, threshold, save_path = False, size_limit =  5 * 10**8):
        """
        Generate a new compression tree using a set of training images.

        This function is used to create a new tree based on the specified set of
        images and the chosen compression threshold. Once generated, the model can
        be saved for future use. Specific details about the training algorithm are
        given in the documention of 'unsupervised.py', but on a practical level it
        is important to remember that the algorithm will follow your threshold
        even if the output feature space becomes larger than your system's memory
        can hold. For this reason, a 'size_limit' parameter is provided so that you
        can set an upper bound on the numer of float32 numbers that can be stored in
        a single array. If this limit is exceeded, training stops and potential OOM
        issues can be avoided. Since training uses Eager Execution while compression
        uses a graph, you cannot immediately deploy a newly generated tree, but must
        instead start a new python session and load the saved arrays using the 'load'
        method.

        Parameters:
            images (ndarray): A set of images to train with.
            threshold (float): A number <= 1 that determines the amount of compression.
            save_path (string): Location to save the model, if 'False' nothing is saved.
            size_limit (integer): Maximum array size allowed during training.
        """
        tf.enable_eager_execution()  #Once this is executed, no static graph can be created or run.
        train_images = images.astype(np.float32)
        if save_path:
            pathlib.Path(save_path).mkdir(parents = True, exist_ok = True)
        trainer = train.Compression_Trainer(self.num_layers)
        for layer in self.nodes:
            for node in layer:
                print('Layer {}, Node {} training'.format(node.layer, node.position))
                isom_array = trainer.train_node(node.layer, node.position, train_images, threshold, size_limit)
                node.isom = tf.constant(isom_array)
                node.output_size = isom_array.shape[0]
                if save_path:
                    isom_path = pathlib.PurePath(save_path).joinpath(node.scope)
                    np.save(str(isom_path), isom_array)
        print('Training Complete')

    def connect_tree(self):
        """
        Connects the nodes of the tree together.

        This function must be called after (not before) the tree has been loaded,
        since it connects the tensors of each individual node together into a single
        graph, which will be used to perform compression. Each node gets connected
        to the two nearest nodes in the previous layer, with the bottom (numbered '0')
        layer being connected directly to the pixels of the image tensor. The final
        pair of nodes become the output of the entire tree, which can then be connected
        to other networks.
        """
        self.images = tf.placeholder(tf.float32, shape = [None, None, None])
        for layer in self.nodes:
            for node in layer:
                left_pos = 2 * node.position  #Factor of 2 since the previous layer is twice as big.
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
        """
        Compress images and output numpy array.

        This function uses the loaded compression tree to compress a batch of
        images and then outputs them as numpy arrays. Since the function runs an
        independent TensorFlow session, it should not be used if you want to connect
        the tree to another graph.

        Parameters:
            images (ndarray): Images to be compressed.

        Returns:
            Tuple of the left output array and right output array respectively.
        """
        with tf.Session() as sess:
            (left_out, right_out) = sess.run((self.left_output, self.right_output), feed_dict = {self.images:images})
        return (left_out, right_out)

    def plot_dimmensions(self):
        """
        Create plot of output sizes at each node.

        This function takes the size of the output array from each node and plots
        them, with each layer on its own set of axes. This is intended to give you
        a sense of how much compression is occuring throughout the tree. A matplotlib
        plot will be displayed when this function is called.
        """
        plot_side = int(np.ceil(np.sqrt(self.num_layers)))  #Side length for square array big enough to hold all layer plots.
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
    """
    Object to hold the compression operations of the tree.

    This class contains the operations and information for a single node of
    the compression tree. All nodes in the tree function the same way, differing
    only in their location and the specific array used for compression.

    Attributes:
        layer (integer): The layer the node is in, with lower layers compressing first.
        position (integer): Location of node within a layer, with lower values
                            compressing the earlier portions of the image.
        scope (string): Variable scope of node for TensorFlow graph.
        isom (tensor): The operation ('isometry') used to compress the inputs.
        output (tensor): Output of the node.
    """
    def __init__(self, layer, position):
        """
        Create empty node object and assign its location.

        Parameters:
            layer (integer): Layer to assign the node.
            position (integer): Position within layer to assign the node.
        """
        self.layer = layer
        self.position = position
        self.scope = '{}_{}'.format(layer, position)

    def create_compression_ops(self, left_in, right_in):
        """
        Create node's tensor operations used for compression.

        This function creates the TensorFlow graph needed for the node to
        compress incoming images. It first performs a kronecker product on the
        two inputs and then multiplies the resulting array by the compression
        operator ('isom') that had been loaded into it.

        Parameters:
            left_in (tensor): The first input for the node.
            right_in (tensor): The second input for the node.
        """
        outer = tf.einsum('ab,ac->abc', left_in, right_in)
        batch_size = tf.shape(left_in)[0]
        kron = tf.reshape(outer, [batch_size, -1])
        self.output = tf.einsum('ij,aj->ai', self.isom, kron)

if __name__ == '__main__':
    #To generate a new tree:
    
    train_path = 'data/8_train'
    tree_path = 'models/64_005'
    num_pixels = 64
    threshold = 0.005

    test_images = data.load_data(train_path)[0]
    tree = Compression_Tree(num_pixels)
    tree.generate(test_images, threshold, save_path = tree_path)
