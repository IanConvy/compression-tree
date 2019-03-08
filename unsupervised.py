import pathlib
import numpy as np
import tensorflow as tf
from tqdm import tqdm

class Compression_Trainer():
    """
    Used to train a compression tree using unsupervised learning.

    This class uses unsupervised learning to determine the correct isometry arrays
    for each node of the compression tree. Given a set of training images, it will
    attempt to compress the feature space such that the features which best
    distinguish the different images are kept, while features that are similar
    across all images are truncated. The degree of truncation is determined by a
    threshold hyperparameter, which will alter the size of the output feature space.
    The training method is deterministic, so trees trained on the same set of images
    with the same threshold will be identical. Further discussion of the compresion
    algorithm can be found in the 'train_node' method docstring. Note that this
    module is not intended to be run independently, but is instead imported and used
    by 'compress.py'.

    Attributes:
        training_nodes (dictionary): Nested dictionary of the training nodes by layer.
    """

    def __init__(self, num_layers):
        """
        Prepare nested dictionary for the training nodes.

        Note that the number of layers must match the size of the images you plan
        to train on.

        Parameters:
            num_layers (integer): Number of layers in the tree to be trained.
        """
        self.training_nodes = {}
        for layer in range(num_layers):
            self.training_nodes[layer] = {}

    def train_node(self, layer, position, train_images, threshold, size_limit):
        """
        Train a node at the given location.

        This function is called to train a node located at a given layer and position
        using the training images and nodes that have already been trained by the same
        class instance. This means that nodes in the same 'causal cone' must be
        trained together, with the lower layers being trained first. The specific
        operations used for training are discussed in the 'Training_Node' class
        docstring, but in general the algorithm proceeds as follows:

        1) The nodes that have already been trained in lower layers are used
        to compress the training images into a new, smaller feature space. When
        training the nodes in the first layer, the orignal feature space is used.

        2) This new feature space is used to construct a covariance matrix for the
        desired node across all of the individual training images in the sample. The
        covariance matrix is then diagonalized, and eigenvectors that correspond to
        smaller eigenvalues are removed from the diagonalizing matrix based on the
        specified threshold. The truncated diagonalizing matrix is the desired result
        of the training, and is used to help generate the new feature space for the next
        layer in the tree. Once all of the nodes in a given layer are trained, the cycle
        can repeat for the next layer.

        Since the training is deterministic, mini-batching is not strictly necessary
        and will not affect the final tensors of the tree. However, for larger image
        sets and lower thresholds, the feature space can easily grow too large to
        fit all of the images into memory at once, leading to OOM errors and possibly
        system crashes. For this reason, the 'size_limit' parameter is provided for
        you to set the maximum number of float32 numbers that are allowed be stored
        in a single array. This value will in turn determine the appropriate batch
        size to use when constructing the covariance matrix. If the feature space
        becomes to large to hold even a single image, training will stop and a warning
        message will be displayed.

        Parameters:
            layer (integer): Layer that you wish to train in.
            position (integer): Position within layer that you wish to train for.
            train_images (ndarray): Batch of images to use for training.
            threshold (float): Value <= 1 that determines degree of compression.
            size_limit (integer): Number of float32 values allowed in single array.

        Returns:
            Numpy array of the truncated diagonalizing matrix for the specified node.
        """
        node = Training_Node(layer, position)
        self.training_nodes[layer][position] = node
        (left_prev_out, right_prev_out) = self.feed_forward(train_images, layer, position, 0, 1)  #Used to determine batch size.
        density_side_len = tf.size(left_prev_out) * tf.size(right_prev_out)
        node.create_initial_params(density_side_len, threshold)
        num_images = train_images.shape[0]
        batch_itr = get_batch_itr(density_side_len, size_limit, num_images)
        if not batch_itr:
            print('Size limit exceeded')
            return
        with tqdm(total = num_images) as pbar:
            for (batch_start, batch_size) in batch_itr:
                pbar.set_postfix({'Batch Size' : batch_size})
                (left_in, right_in) = self.feed_forward(train_images, layer, position, batch_start, batch_size)  #Step 1 in docstring.
                node.add_batch(left_in, right_in, batch_size)  #Images are added in mini-batches to avoid OOM.
                pbar.update(batch_size)
        node.diagonalize(num_images)  #Step 2 in doctring.
        print('Output Size = {}'.format(node.output_size))
        return node.isom_var.numpy()

    def feed_forward(self, images, layer, position, batch_start, batch_size):
        """
        Create compressed feature space using trained nodes.

        This recursive function takes a set of images and compresses them using
        the nodes that were trained from previous calls to 'train_node', returning
        the pair of outputs that a node would recieve from the tree if located at
        the specified layer and position. Effectively, this function recursively
        triggers the 'causal cone' of nodes down the tree that are necessary to
        create the output at some point higher up the tree. When the function
        reaches the bottom layer, it returns the feature values of the image pixels
        and stops its recursion, at which point those pixel values are compressed up
        through the nodes in the causal cone to generate the final pair of outputs
        at the specified location.

        Parameters:
            images (ndarray): Images to be compressed through the casual cone.
            layer (integer): Layer to get the output in.
            position (integer): Position within layer to get the output.
            batch_start (integer): Index from which to start a mini-batch.
            batch_size (integer): Number of images to compress in mini-batch.

        Returns:
            Tuple containing the left/right outputs of the tree at specified location.
        """
        left_pos = 2 * position  #Factor of 2 since preceding layer is twice as big.
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
    """
    Object to hold the training operations of the tree.

    This class contains the operations and information for a single node during
    and after its training. The node must be able to:
        1) Create a covariance matrix from multiple mini-batches of compressed images
        2) Diagonalize that matrix
        3) Truncate the eigenvectors appropriately to create a compression operator
        4) Use that operator to compress images
    The specific TensorFlow operations that go into each of these steps are described
    in the docstrings of their corresponding functions.

    Attributes:
        scope (string): Varaible scope to be used in the graph.
        threshold (float): Value <= 1 that determines degree of compression.
        density_var (tensor): Tensor to collect mini-batches one-by-one.
        isom_var (tensor): Tensor used by the node to compress images.
        output_size (integer): Size of the output feature space.
    """

    def __init__(self, layer, position):
        """
        Create empty node object.

        Parameters:
            layer (integer): Layer node is in.
            position (integer): Position within layer for node.
        """
        self.scope = '{}_{}'.format(layer, position)

    def create_initial_params(self, density_side_len, threshold):
        """
        Creates the threshold and density_var attributes.

        This function is seperated because it only needs to be called once before
        the mini-batch loop. The size of the density variable can only be determined
        once nodes in the previous layer have been trained, which is why this module
        is run using Eager Execution.

        Parameters:
            density_side_len (integer): Size of the input space for the node.
            threshold (float): Value <= 1 that determines degree of compression.
        """
        self.threshold = threshold
        with tf.variable_scope(self.scope):
            self.density_var = tf.get_variable('Batch_Density', initializer = tf.zeros([density_side_len, density_side_len]))

    def add_batch(self, left_in, right_in, batch_size):
        """
        Adds mini_batches to a cumulative sum.

        This function corresponds to step 1 in the class docstring. It creates
        operators that first take the kroncker product of the node's two input
        arrays, and then takes the outer product of this combined array to create
        what is effectively a mini-batch of image density matrices. The arrays in
        this mini-batch are then added to a variable that persists across mini-batch
        session runs, allowing for the sum to be accumulated across all of the training
        images.

        Parameters:
            left_in (tensor): Left input going into node from the previous layer.
            right_in (tensor): Right input going into node from the previous layer.
        """
        outer = tf.einsum('ab,ac->abc', left_in, right_in)
        kron = tf.reshape(outer, [batch_size, -1])
        density = tf.einsum('ab,ac->abc', kron, kron)
        batch_density = tf.reduce_sum(density, axis = 0)
        tf.assign_add(self.density_var, batch_density)

    def diagonalize(self, num_total_images):
        """
        Creates and diagonalizes the node's covariance matrix.

        This function corresponds to the last bits of step 1 and all of steps 2
        and 3 in the class docstring. It takes the variable that was created across
        the 'add_batch' method calls and forms the covariance matrix by averaging it
        with respect to the total number of training images. This matrix is then
        traced and diagonalized. To determine which eigenvalues (and thus eigenvectors)
        will be truncated, the eigenvalues are summed in ascending order until the
        ratio of the sum to the trace exceeds the specified threshold, at which
        point all of the eigenvectors corresponding to the summed eigenvalues are
        removed from the diagonalizing matrix. This truncated tensor is then stored
        as a variable.

        Parameters:
            num_total_images (integer): Number of images in the entire training set.
        """
        herm_matrix = self.density_var / num_total_images
        trace = tf.trace(herm_matrix)
        (eig_values, eig_vects) = tf.linalg.eigh(herm_matrix)  #Eig values are sorted in ascending order by default.
        cum_eigs = tf.cumsum(eig_values)
        ratios = cum_eigs / trace
        thresh_test = tf.cast(ratios <= self.threshold, tf.int32)
        trunc_index = tf.reduce_sum(thresh_test)
        trunc_isom = tf.transpose(eig_vects[:, trunc_index:])
        with tf.variable_scope(self.scope):
            self.isom_var = tf.get_variable('Isometry', initializer = trunc_isom)
        self.output_size = int(tf.shape(self.isom_var)[0])

    def compress(self, left_in, right_in, batch_size):
        """
        Compresses a mini-batch of images.

        This function corresponds to step 4 in the class docstring. It takes the
        node's two input arrays and combines them together into a kronecker product,
        then acts on them with the truncated matrix created from the 'diagonalize'
        method. These operations are only used after the node is trained, to create
        the feature space that will be used to train nodes in the higher layers.

        Parameters:
            left_in (tensor): Left input going into node from the previous layer.
            right_in (tensor): Right input going into node from the previous layer.
            batch_size (integer): Size of the mini-batches of the inputs.

        Returns:
            Tensor containing the compressed mini-batch.
        """
        outer = tf.einsum('ab,ac->abc', left_in, right_in)
        kron = tf.reshape(outer, [batch_size, -1])
        compr_out = tf.einsum('ij,aj->ai', self.isom_var, kron)
        return compr_out

def get_batch_itr(density_side_len, limit, num_images):
    """
    Create iterator for mini-batches.

    This generator creates and returns an iterator that yields indices which can
    be used to slice off mini-batch arrays of a limited size. This is useful for
    avoiding OOM errors if the tree ends up creating large feature spaces. If a
    mini-batch size of 1 still exceeds the specified limit, the generator returns
    an empty list. The mini-batch size can be any integer less than the total number
    of images, with the final mini-batch possibly containing fewer images if the
    batch size is not a factor of the numer of images.

    Parameters:
            density_side_len (integer): Size of feature space.
            limit (integer): Max number of elements allowed in an array.
            num_images (integer): Total number of images to be mini-batched.

    Yields:
        Tuple containing the starting index and size of the mini-batch.
    """
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
