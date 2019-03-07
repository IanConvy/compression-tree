import pickle as pk
import numpy as np
import tensorflow as tf
import skimage.transform
import matplotlib.pyplot as plt

class DataGenerator:
    """
    A class to create and manipulate the MNIST dataset.

    Attributes:
        train_images (ndarray): batched MNIST training images
        test_images (ndarray): batched MNIST test images
        train_labels(ndarray): batched MNIST train label integers
        test_lables(ndarray): batched MNIST test label integers
    """

    def __init__(self):
        """Creates a copy of the MNIST data to be futher processed."""
        mnist_data = tf.keras.datasets.mnist.load_data()
        self.train_images = mnist_data[0][0]
        self.train_labels = mnist_data[0][1]
        self.test_images = mnist_data[1][0]
        self.test_labels = mnist_data[1][1]

    def scale_images(self, new_shape):
        """
        Resizes the images using Scikit-Image.

        This function should be run before any featurization. It will
        renormalize the grayscale values into the interval [0, 1], and
        apply anti_aliasing when needed.

        Parameters:
            new_shape (sequence): Desired image shape.
        """
        self.train_images = resize_images(self.train_images, new_shape)
        self.test_images = resize_images(self.test_images, new_shape)

    def featurize(self, feat_funct):
        """
        Applies a featurization to the MNIST images.

        This function should only be run after you have performed any
        desired resizing, as it will likely alter the shape of the images.
        The featurization functions in this module can all be applied via
        this method.

        Parameters:
            feat_funct (callable): A function that can be applied to a batch.
        """
        self.train_images = feat_funct(self.train_images)
        self.test_images = feat_funct(self.test_images)

    def export(self, path):
        """
        Pickles the training and test data.

        This function should be run after all desired processing is complete.
        It saves the training data and test data as seperate pickled tuples of
        images and labels respectively.

        Parameters:
            path (string): File path to save the data.
        """
        train_dest = path + '_train'
        test_dest = path + '_test'
        save_data(self.train_images, self.train_labels, train_dest)
        save_data(self.test_images, self.test_labels, test_dest)

def select_digits(images, labels, digits):
    """
    Returns data for only the specified digits.

    This function takes the image batch and label batch and removes all image/label
    pairs that are not in the 'digits' parameter. The function will not work on one-hot
    labels.

    Parameters:
        images (ndarray): Batch of images to be filtered.
        labels (ndarray): Vector of integer labels matched to the image batch.
        digits (sequence): Digits to keep.

    Returns:
        Filtered image and label batches as a tuple.
    """
    cumulative_test = labels == digits[0]
    for digit in digits[1:]:
        digit_test = labels == digit
        cumulative_test = np.logical_or(digit_test, cumulative_test)
    valid_images = images[cumulative_test]
    valid_labels = labels[cumulative_test]
    return (valid_images, valid_labels)

def resize_images(images, shape):
    """
    Uses Scikit-Image to resize image batch.

    Parameters:
        images (ndarray): Image batch to be resized.

    Returns:
        ndarray: Resized image batch.
    """
    num_images = images.shape[0]
    new_images_shape = (num_images, shape[0], shape[1])
    new_images = skimage.transform.resize(
        images,
        new_images_shape,
        anti_aliasing = True,
        mode = 'constant')
    return new_images

def batch_generator(images, labels, batch_size):
    """
    Returns an iterator dividing the data into mini-batches in a random order.

    Parameters:
        images (ndarray): Batch of images to divide.
        labels (ndarray): Batch of labels to divide.

    Yields:
        Tuple of image mini-batch and label mini-batch.
    """
    num_images = images.shape[0]
    random_perm = np.random.permutation(num_images)
    randomized_images = images[random_perm]
    randomized_labels = labels[random_perm]
    for i in range(0, num_images, batch_size):
        batch_images = randomized_images[i : i + batch_size]
        batch_labels = randomized_labels[i : i + batch_size]
        yield batch_images, batch_labels

def flatten_images(images):
    """
    Flattens a batch of image matrices into a batch of vectors.

    Parameters:
        images (ndarray): Batch of images to be flattened.
    """
    num_images = images.shape[0]
    flattened_image = np.reshape(images, [num_images, -1])
    return flattened_image

def poly_feat(images):
    """
    Featurizes the image pixels into the form [1, pixel].

    This featurization function takes an image batch, flattens it, and then
    transforms each image vector into a two-row matrix array with each column
    having a 1 in the first row and the pixel value in the second row.

    Parameters:
        images (ndarray): Image batch to featurize.

    Returns:
        Image batch with featurized images.
    """
    flat_images = flatten_images(images)
    prep_images = np.expand_dims(flat_images, axis = 1)
    ones = np.ones_like(prep_images)
    feature_array = np.concatenate([ones, prep_images], axis = 1)
    return feature_array

def trig_feat(images):
    """
    Featurizes the image pixels into the form [cos(pixel), sin(pixel)].

    This featurization function takes an image batch, flattens it, and then
    transforms each image vector into a two-row matrix array with each column
    having cos(pixel) in the first row and sin(pixel) in the second row.

    Parameters:
        images (ndarray): Image batch to featurize.

    Returns:
        Image batch with featurized images.
    """
    flat_images = flatten_images(images)
    (num_images, num_pixels) = flat_images.shape
    prep_axes = np.reshape(flat_images, (num_images, num_pixels, 1))
    pix_copy = np.tile(prep_axes, [1, 1, 2])
    pix_copy[:, :, 0] = np.cos(pix_copy[:, :, 0] * np.pi/2)
    pix_copy[:, :, 1] = np.sin(pix_copy[:, :, 1] * np.pi/2)
    return pix_copy

def split_data(images, labels, split):
    """
    Splits data into two batches using a randomized order.

    Parameters:
        images (ndarray): Image batch to split
        labels (ndarray): Label batch to split
        split (integer): Fraction of data in the first batch.

    Returns:
        Tuple of tuples, inner tuples are image/label batches and outer tuple
        is left/right split.
    """
    num_images = images.shape[0]
    random_perm = np.random.permutation(num_images)
    randomized_images = images[random_perm]
    randomized_labels = labels[random_perm]
    split_point = int(split * num_images)
    left_split = (images[:split_point], labels[:split_point])
    right_split = (images[split_point:], labels[split_point:])
    return (left_split, right_split)

def one_hot(labels, num_classes = 10):
    """
    Transforms integer labels to one-hot vectors.

    This function takes a vector of integer labels and expands it into
    a matrix with a number of columns equal to the number of classes. Each row
    will have a value of 1 in the column corresponding to the label, and zeros
    otherwise.

    Parameters:
        labels (ndarray): Vector of labels to be transformed.
        num_classes (integer): Number of classes for the labels

    Returns:
        Label batch with one-hot on the second axis.
    """
    blank = np.zeros(labels.size * num_classes)
    multiples = np.arange(labels.size)
    index_shift = num_classes * multiples
    new_indices = labels + index_shift
    blank[new_indices] = 1
    matrix_blank = np.reshape(blank, [labels.size, num_classes])
    return matrix_blank

def get_data(data_path, digits, val_split = 1):
    """
    Loads data, selects digits, and performs validation split.

    This function is a convenient way to load saved data, select specific digits,
    and then split off a fraction of the data to hold out for validation.

    Parameters:
        data_path (string): path to data, but does not include the '_train' or
                            '_test' suffix.
        digits (sequence): Digits to keep
        val_split: Fraction of data to hold out for validation.
    """
    train_raw = load_data(data_path + '_train')
    test_raw = load_data(data_path + '_test')
    (train_images, train_labels) = select_digits(*train_raw, digits)
    (test_images, test_labels) = select_digits(*test_raw, digits)
    (true_train_data, val_data) = split_data(train_images, train_labels, val_split)
    return (true_train_data, val_data, (test_images, test_labels))

def save_data(images, labels, path):
    """
    Pickles data as images/labels tuple.

    Parameters:
        images (ndarray): Images to be saved.
        labels (ndarray): Labels to be saved.
        path (string): Location to save.
    """
    dest = open(path, 'wb')
    data = (images, labels)
    pk.dump(data, dest)

def load_data(path):
    """
    Loads pickled data.

    Parameters:
        path (string): File to be loaded.
    """
    dest = open(path, 'rb')
    data = pk.load(dest)
    return data
