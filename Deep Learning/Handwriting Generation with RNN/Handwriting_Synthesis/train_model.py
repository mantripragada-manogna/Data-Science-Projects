import os
import tensorflow as tf
from ModelGraph import create_graph
from batch_generator import BatchGenerator
import datetime

def main():
    seq_len = 256
    batch_size = 64
    epochs = 30
    batches_per_epoch = 1000

    batch_generator = BatchGenerator(batch_size, seq_len)
    g, vs = create_graph(batch_generator.num_letters, batch_size)

    with tf.Session(graph=g) as sess:
        model_saver = tf.train.Saver(max_to_keep=2)
        sess.run(tf.global_variables_initializer())
        model_path = get_model_path()

        summary_writer = tf.summary.FileWriter(model_path, graph=g, flush_secs=10)
        summary_writer.add_session_log(tf.SessionLog(status=tf.SessionLog.START), global_step=0)
        for e in range(epochs):
            print('\n{} : Epoch {}'.format(datetime.datetime.now().time(),e))
            for b in range(1, batches_per_epoch + 1):
                coordinates, labels, reset, to_reset = batch_generator.next_batch()
                if to_reset:
                    sess.run(vs.reset_states, feed_dict={vs.reset: reset})
                loss, s, _ = sess.run([vs.loss, vs.summary, vs.train_step],
                                      feed_dict={vs.coordinates: coordinates, vs.sequence: labels})
                summary_writer.add_summary(s, global_step=e * batches_per_epoch + b)
                print('\r[{:5d}/{:5d}] loss = {}'.format(b, batches_per_epoch, loss), end='')

            model_saver.save(sess, os.path.join(model_path, 'models', 'model'), global_step=e)


def get_model_path():

    idx = 0
    path = os.path.join('summary', 'experiment-{}')
    while os.path.exists(path.format(idx)):
        idx += 1
    path = path.format(idx)
    os.makedirs(os.path.join(path, 'models'))

    return path


if __name__ == '__main__':
    main()
