import tensorflow as tf
import numpy as np
from SoftWindowLayer import SoftWindow
from LSTMLayers import LSTMLayer
from MDNLayer import MixtureLayer
from collections import namedtuple

epsilon = 1e-8


def create_graph(num_letters, batch_size,
                 num_units=400, window_mixtures=10, output_mixtures=20):

    graph = tf.Graph()
    with graph.as_default():
        coordinates = tf.placeholder(tf.float32, shape=[None, None, 3])
        sequence = tf.placeholder(tf.float32, shape=[None, None, num_letters])
        reset = tf.placeholder(tf.float32, shape=[None, 1])
        bias = tf.placeholder_with_default(tf.zeros(shape=[]), shape=[])

        def create_model(synthesize=None):
            input_coordinates = coordinates[:, :-1, :]
            output_coordinates = coordinates[:, 1:, :]

            _batch_size = 1 if synthesize else batch_size
            if synthesize:
                input_coordinates = coordinates

            with tf.variable_scope('model', reuse=synthesize):
                window = SoftWindow(num_mix=window_mixtures, labels=sequence, num_chars=num_letters)
                rnn_model = LSTMLayer(num_letters=num_letters, window_layer=window, batch_size=_batch_size, input_size=3)
                reset_states = tf.group(*[state.assign(state * reset) for state in rnn_model.states])

                outs, states = tf.nn.dynamic_rnn(rnn_model, input_coordinates, initial_state=rnn_model.states)

                mdn_layer = MixtureLayer(input_size=num_units, output_size=2, num_mixtures=output_mixtures)

                with tf.control_dependencies([sp.assign(sc) for sp, sc in zip(rnn_model.states, states)]):
                    with tf.name_scope('prediction'):
                        outs = tf.reshape(outs, [-1, num_units])
                        e, pi, mu1, mu2, std1, std2, rho = mdn_layer(outs, bias)

                    with tf.name_scope('loss'):
                        coords = tf.reshape(output_coordinates, [-1, 3])
                        xs, ys, es = tf.unstack(tf.expand_dims(coords, axis=2), axis=1)

                        mrho = 1 - tf.square(rho)
                        xms = (xs - mu1) / std1
                        yms = (ys - mu2) / std2
                        z = tf.square(xms) + tf.square(yms) - 2. * rho * xms * yms
                        n = 1. / (2. * np.pi * std1 * std2 * tf.sqrt(mrho)) * tf.exp(-z / (2. * mrho))
                        ep = es * e + (1. - es) * (1. - e)
                        rp = tf.reduce_sum(pi * n, axis=1)

                        loss = tf.reduce_mean(-tf.log(rp + epsilon) - tf.log(ep + epsilon))

                    if synthesize:
                        for param in [('coordinates', coordinates), ('sequence', sequence),
                                      ('bias', bias), ('e', e), ('pi', pi),
                                      ('mu1', mu1), ('mu2', mu2), ('std1', std1), ('std2', std2),
                                      ('rho', rho), ('phi', rnn_model.prev_phi),
                                      ('window', rnn_model.states[-3]), ('kappa', rnn_model.states[-2]),
                                      ('finish', rnn_model.states[-1]), ('zero_states', rnn_model.zero_states)]:
                            tf.add_to_collection(*param)

                with tf.name_scope('training'):
                    steps = tf.Variable(0.)
                    learning_rate = tf.train.exponential_decay(0.001, steps, staircase=True, decay_steps=10000, decay_rate=0.5)
                    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                    grad, var = zip(*optimizer.compute_gradients(loss))
                    grad, _ = tf.clip_by_global_norm(grad, 3.)
                    train_step = optimizer.apply_gradients(zip(grad, var), global_step=steps)

                with tf.name_scope('summary'):
                    # TODO: add more summaries
                    summary = tf.summary.merge([
                        tf.summary.scalar('loss', loss)
                    ])

                return namedtuple('Model', ['coordinates', 'sequence', 'reset_states', 'reset', 'loss', 'train_step', 'learning_rate', 'summary'])\
                    (coordinates, sequence, reset_states, reset, loss, train_step, learning_rate, summary)
        train_model = create_model(synthesize=None)
        _ = create_model(synthesize=True)

    return graph, train_model
