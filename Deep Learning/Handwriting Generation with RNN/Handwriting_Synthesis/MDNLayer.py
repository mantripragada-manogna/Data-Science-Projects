import tensorflow as tf


class MixtureLayer(object):
    """
        This layer serves as the last output layer for the Synthesis model.
        Its takes in the input from the LSTM Layer and outputs a probablity distributions
        that represents the placement of the co-ordinates for the points and the end of stroke.

        Below are the components that are returned:
        1) e (eta) - End of stroke probability
        2) 2 sets of means (mu) - Set of means for the Gaussian distributions representing
                                  the two co-ordinates of a point
        3) 2 set of std div (sigma) - Set of standard deviation for the Gaussian distributions representing
                                      the two co-ordinates of a point
        4) correlation (rho) - Represents the co-relations among the co-ordinates and is required for
                               calculating the co-variance in a multivariate distribution
        5) mixture weights (pi) - Mixture weights for each mixture components and a re used for
                                  calculating the loss functions and in-turn training the model.

    """
    def __init__(self, input_size, output_size, num_mixtures):
        self.input_size = input_size
        self.output_size = output_size
        self.num_mixtures = num_mixtures

    def __call__(self, inputs, bias=0., reuse=None):
        with tf.variable_scope('mixture_output', reuse=reuse):
            e = tf.layers.dense(inputs, 1,
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.075), name='e')
            pi = tf.layers.dense(inputs, self.num_mixtures,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.075), name='pi')
            mu1 = tf.layers.dense(inputs, self.num_mixtures,
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.075), name='mu1')
            mu2 = tf.layers.dense(inputs, self.num_mixtures,
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.075), name='mu2')
            std1 = tf.layers.dense(inputs, self.num_mixtures,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.075), name='std1')
            std2 = tf.layers.dense(inputs, self.num_mixtures,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.075), name='std2')
            rho = tf.layers.dense(inputs, self.num_mixtures,
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.075), name='rho')

            return tf.nn.sigmoid(e),\
                   tf.nn.softmax(pi * (1. + bias)), \
                   mu1, mu2, \
                   tf.exp(std1 - bias), tf.exp(std2 - bias), \
                   tf.nn.tanh(rho)
