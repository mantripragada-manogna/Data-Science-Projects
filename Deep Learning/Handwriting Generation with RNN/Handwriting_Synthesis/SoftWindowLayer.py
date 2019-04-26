import tensorflow as tf
import numpy as np


class SoftWindow(object):
    """
        This Soft windows guides the placement of the strokes and the characters
        The network learns how far to slide each window at each step, rather than an absolute location.
        Using offsets was essential to getting the network to align the text with the pen trace.

        This window is defined by three parameters
        alpha(a) - It controls the importance of the window
        beta(b) - It controls the window of the window
        kappa (k) - It controls the location of the window

        It takes the following inputs:
        lables - One hot encoded lables
        Number of Mixtures - The number of K Gaussian functions
        Number of characters - Needed to determine the size of the output
        (Output window will be of the dimension - [Number Of characters X Number of Mixtures X 1]
    """
    def __init__(self, num_mix, labels, num_chars):
        """
        :param num_mix: number of mixtures
        :param labels: one hot encoded labels
        :param num_chars: total number of characters
        """
        self.labels = labels
        self.labels_size = tf.shape(labels)[1]
        self.num_mix = num_mix
        self.num_chars = num_chars
        self.u_indexes = -tf.expand_dims(
            tf.expand_dims(tf.range(0., tf.cast(self.labels_size, dtype=tf.float32)), axis=0), axis=0)

    def __call__(self, inputs, k, reuse=None):
        """

        :param inputs: Here Input will be the output from the first LSTM layer
        :param k: It is the Kappa value controls the location of the window
        :param reuse: Determines if the window variable needs to be reused. (Always None)
        :return: A tuple of four:
                1)  Window tensor to be forwarded to the next LSTM layer
                2)  Kappa values to be carried forward
                3)  Window weights phi
                4)  Finish variables that identifies if the sequence generation is completed.
        """
        with tf.variable_scope('window', reuse=reuse):
            alpha = tf.layers.dense(inputs, self.num_mix, activation=tf.exp,
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.075),
                                    name='alpha')
            beta = tf.layers.dense(inputs, self.num_mix, activation=tf.exp,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.075),
                                   name='beta')
            kappa = tf.layers.dense(inputs, self.num_mix, activation=tf.exp,
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.075),
                                    name='kappa')

            a = tf.expand_dims(alpha, axis=2)
            b = tf.expand_dims(beta, axis=2)
            k = tf.expand_dims(k + kappa, axis=2)

            phi = tf.exp(-np.square(self.u_indexes + k) * b) * a
            phi = tf.reduce_sum(phi, axis=1, keep_dims=True)

            finish = tf.cast(phi[:, 0, -1] > tf.reduce_max(phi[:, 0, :-1], axis=1), tf.float32)

            window = tf.squeeze(tf.matmul(phi, self.labels), axis=1)

            return window,\
                   tf.squeeze(k, axis=2), \
                   tf.squeeze(phi, axis=1),\
                   tf.expand_dims(finish, axis=1)

    @property
    def output_size(self):
        return [self.num_chars, self.num_mix, 1]
