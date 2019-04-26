import tensorflow as tf


class LSTMLayer(tf.nn.rnn_cell.RNNCell):
    """
        This is a custom LSTM layer which includes one Soft Window between the first and
        the second window to guide model the stroke and the characters alignment.

        We will create a layer of 3 LSTM cells each containing 400 cells
    """
    def __init__(self, num_letters, batch_size, window_layer, input_size, num_layers=3, num_units=400):
        """
        :param num_letters: The number of letters in the translation
        :param batch_size: The batch size for training
        :param window_layer: The Soft window
        :param num_layers: The Number of LSTM Cells
        :param num_units: The number of cells inside a single LSTM
        """
        super(LSTMLayer, self).__init__()
        self.num_layers = num_layers
        self.num_units = num_units
        self.num_letters = num_letters
        self.window_layer = window_layer
        self.input_size = input_size
        self.prev_phi = None

        with tf.variable_scope('rnn', reuse=None):
            self.lstmCells = [tf.nn.rnn_cell.LSTMCell(num_units)
                              for i in range(num_layers)]
            self.states = [tf.Variable(tf.zeros([batch_size, s]), trainable=False)
                           for s in self.state_size]
            self.zero_states = tf.group(*[sp.assign(sc)
                                          for sp, sc in zip(self.states,
                                                            self.zero_state(batch_size, dtype=tf.float32))])

    def call(self, inputs, state, **kwargs):
        """

        :param inputs: The input will be the co-ordinates for the stroke
        :param state: The state of the previous LSTM
        :param kwargs: not used
        :return: Returns a tuple of two components output and
                the state of the last LSTM cell + the window output
        """
        window, k, finish = state[-3:]
        out_state = []
        prev_out = []

        for i in range(self.num_layers):
            x = tf.concat([inputs, window] + prev_out, axis=1)
            with tf.variable_scope('LSTM_LAYER_{}'.format(i)):
                out, s = self.lstmCells[i](x, tf.nn.rnn_cell.LSTMStateTuple(state[2 * i], state[2 * i + 1]))
                prev_out = [out]
            out_state += [*s]

            if i == 0:
                window, k, self.prev_phi, finish = self.window_layer(out, k)

        return out, out_state + [window, k, finish]

    @property
    def state_size(self):
        return [self.num_units] * self.num_layers * 2 + self.window_layer.output_size

    @property
    def output_size(self):
        return [self.num_units]
