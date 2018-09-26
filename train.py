from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import input_data
import models
from tensorflow.python.platform import gfile

FLAGS = None


def main(_):

  tf.logging.set_verbosity(tf.logging.INFO)

  #NOVA TENSORFLOW SESIJA
  sess = tf.InteractiveSession()

  model_settings = models.prepare_model_settings(
      len(input_data.prepare_words_list(FLAGS.wanted_words.split(','))),
      FLAGS.sample_rate, 
      FLAGS.clip_duration_ms, 
      FLAGS.window_size_ms,
      FLAGS.window_stride_ms, 
      FLAGS.feature_bin_count, 
      FLAGS.preprocess)
  audio_processor = input_data.AudioProcessor(
      FLAGS.data_dir,
      FLAGS.silence_percentage, 
      FLAGS.unknown_percentage,
      FLAGS.wanted_words.split(','), 
      FLAGS.validation_percentage,
      FLAGS.testing_percentage, 
      model_settings, 
      FLAGS.summaries_dir)
  fingerprint_size = model_settings['fingerprint_size']
  label_count = model_settings['label_count']
  time_shift_samples = int((FLAGS.time_shift_ms * FLAGS.sample_rate) / 1000)


  training_steps_list = list(map(int, FLAGS.how_many_training_steps.split(',')))
  learning_rates_list = list(map(float, FLAGS.learning_rate.split(',')))
  if len(training_steps_list) != len(learning_rates_list):
    raise Exception(
        '--how_many_training_steps and --learning_rate must be equal length '
        'lists, but are %d and %d long instead' % (len(training_steps_list), len(learning_rates_list)))

  input_placeholder = tf.placeholder(tf.float32, [None, fingerprint_size], name='fingerprint_input')
  if FLAGS.quantize:
    # TODO
    if FLAGS.preprocess == 'average':
      fingerprint_min = 0.0
      fingerprint_max = 2048.0
    elif FLAGS.preprocess == 'mfcc':
      fingerprint_min = -247.0
      fingerprint_max = 30.0
    else:
      raise Exception('Unknown preprocess mode "%s" (should be "mfcc" or'
                      ' "average")' % (FLAGS.preprocess))
    fingerprint_input = tf.fake_quant_with_min_max_args(
        input_placeholder, fingerprint_min, fingerprint_max)
  else:
    fingerprint_input = input_placeholder

  logits, dropout_prob = models.create_model(fingerprint_input, model_settings, FLAGS.model_architecture, is_training=True)

  ground_truth_input = tf.placeholder(
      tf.int64, [None], name='groundtruth_input')

  control_dependencies = []
  if FLAGS.check_nans:
    checks = tf.add_check_numerics_ops()
    control_dependencies = [checks]

  #INICIJALIZACIJA MEHANIZMA ZA TRENIRANJE
  with tf.name_scope('cross_entropy'):
    cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(labels=ground_truth_input, logits=logits)

  if FLAGS.quantize:
    tf.contrib.quantize.create_training_graph(quant_delay=0)

  with tf.name_scope('train'), tf.control_dependencies(control_dependencies):
    learning_rate_input = tf.placeholder(
        tf.float32, [], name='learning_rate_input')
    train_step = tf.train.GradientDescentOptimizer(
        learning_rate_input).minimize(cross_entropy_mean)

  predicted_indices = tf.argmax(logits, 1)
  correct_prediction = tf.equal(predicted_indices, ground_truth_input)
  confusion_matrix = tf.confusion_matrix(
      ground_truth_input, predicted_indices, num_classes=label_count)
  evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  with tf.get_default_graph().name_scope('eval'):
    tf.summary.scalar('cross_entropy', cross_entropy_mean)
    tf.summary.scalar('accuracy', evaluation_step)

  global_step = tf.train.get_or_create_global_step()
  increment_global_step = tf.assign(global_step, global_step + 1)

  saver = tf.train.Saver(tf.global_variables())

  #ZABELEZI IZVESTAJ U retrain_logs
  merged_summaries = tf.summary.merge_all(scope='eval')
  train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
  validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')


  tf.global_variables_initializer().run()
  if FLAGS.start_checkpoint:
    models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)
    start_step = global_step.eval(session=sess)
  else:
    start_step = 1


  tf.logging.info('Training from step: %d ', start_step)
  tf.train.write_graph(sess.graph_def, FLAGS.train_dir, FLAGS.model_architecture + '.pbtxt')
  with gfile.GFile(os.path.join(FLAGS.train_dir, FLAGS.model_architecture + '_labels.txt'), 'w') as f:
    f.write('\n'.join(audio_processor.words_list))



  #OBUCAVANJE
  training_steps_max = np.sum(training_steps_list)
  for training_step in xrange(start_step, training_steps_max + 1):

    training_steps_sum = 0
    for i in range(len(training_steps_list)):
      training_steps_sum += training_steps_list[i]
      if training_step <= training_steps_sum:
        learning_rate_value = learning_rates_list[i]
        break

    #DOHVATI ZVUK ZA OBUCAVANJE
    train_fingerprints, train_ground_truth = audio_processor.get_data(
        FLAGS.batch_size, 0, model_settings, FLAGS.background_frequency,
        FLAGS.background_volume, time_shift_samples, 'training', sess)
    # Run the graph with this batch of training data.
    train_summary, train_accuracy, cross_entropy_value, _, _ = sess.run(
        [merged_summaries, evaluation_step, cross_entropy_mean, train_step, increment_global_step,],
        feed_dict={
            fingerprint_input: train_fingerprints,
            ground_truth_input: train_ground_truth,
            learning_rate_input: learning_rate_value,
            dropout_prob: 0.5
        })
    train_writer.add_summary(train_summary, training_step)
    tf.logging.info('Step #%d: rate %f, accuracy %.1f%%, cross entropy %f' %
                    (training_step, learning_rate_value, train_accuracy * 100, cross_entropy_value))
    is_last_step = (training_step == training_steps_max)
    if (training_step % FLAGS.eval_step_interval) == 0 or is_last_step:
      set_size = audio_processor.set_size('validation')
      total_accuracy = 0
      total_conf_matrix = None
      for i in xrange(0, set_size, FLAGS.batch_size):
        validation_fingerprints, validation_ground_truth = (
            audio_processor.get_data(FLAGS.batch_size, i, model_settings, 0.0, 0.0, 0, 'validation', sess))
        # Run a validation step and capture training summaries for TensorBoard
        # with the `merged` op.
        validation_summary, validation_accuracy, conf_matrix = sess.run([merged_summaries, evaluation_step, confusion_matrix],
            feed_dict={fingerprint_input: validation_fingerprints, ground_truth_input: validation_ground_truth, dropout_prob: 1.0})
        validation_writer.add_summary(validation_summary, training_step)
        batch_size = min(FLAGS.batch_size, set_size - i)
        total_accuracy += (validation_accuracy * batch_size) / set_size
        if total_conf_matrix is None:
          total_conf_matrix = conf_matrix
        else:
          total_conf_matrix += conf_matrix

      tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
      tf.logging.info('Step %d: Validation accuracy = %.1f%% (N=%d)' % (training_step, total_accuracy * 100, set_size))


    if(training_step % FLAGS.save_step_interval == 0 or training_step == training_steps_max):
      checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.model_architecture + '.ckpt')
      tf.logging.info('Saving to "%s-%d"', checkpoint_path, training_step)
      saver.save(sess, checkpoint_path, global_step=training_step)



  set_size = audio_processor.set_size('testing')
  tf.logging.info('set_size=%d', set_size)
  total_accuracy = 0
  total_conf_matrix = None
  for i in xrange(0, set_size, FLAGS.batch_size):
    test_fingerprints, test_ground_truth = audio_processor.get_data(FLAGS.batch_size, i, model_settings, 0.0, 0.0, 0, 'testing', sess)
    test_accuracy, conf_matrix = sess.run([evaluation_step, confusion_matrix],
        feed_dict={fingerprint_input: test_fingerprints, ground_truth_input: test_ground_truth, dropout_prob: 1.0})
    batch_size = min(FLAGS.batch_size, set_size - i)
    total_accuracy += (test_accuracy * batch_size) / set_size
    if total_conf_matrix is None:
      total_conf_matrix = conf_matrix
    else:
      total_conf_matrix += conf_matrix

  tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
  tf.logging.info('Final test accuracy = %.1f%% (N=%d)' % (total_accuracy * 100, set_size))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/home/master/Desktop/py/speech_recognition/speech_dataset/')
  parser.add_argument('--background_volume', type=float, default=0.1)
  #PROCENAT UZORAKA KOJI IMAJU POZADINSKU BUKU U SKUPU ZA OBUCAVANJE
  parser.add_argument('--background_frequency', type=float, default=0.8)
  #PROCENAT UZORAKA TISINE U SKUPU ZA OBUCAVANJE
  parser.add_argument('--silence_percentage', type=float, default=10.0)
  #PROCENAT UZORAKA KOJI PREDSTAVLJAJU NEPOZNATE RECI U SKUPU ZA OBUCAVANJE
  parser.add_argument('--unknown_percentage', type=float, default=10.0)
  parser.add_argument('--time_shift_ms', type=float, default=100.0)
  parser.add_argument('--testing_percentage', type=int, default=10)
  parser.add_argument('--validation_percentage', type=int, default=10)
  parser.add_argument('--sample_rate', type=int, default=16000)
  parser.add_argument('--clip_duration_ms', type=int, default=1000)
  parser.add_argument('--window_size_ms', type=float, default=30.0)
  parser.add_argument('--window_stride_ms', type=float, default=10.0)
  parser.add_argument('--feature_bin_count', type=int, default=40)
  parser.add_argument('--how_many_training_steps', type=str, default='15000,3000')
  parser.add_argument('--eval_step_interval', type=int, default=400)
  parser.add_argument('--learning_rate', type=str, default='0.001,0.0001')
  parser.add_argument('--batch_size', type=int, default=100)
  parser.add_argument('--summaries_dir', type=str, default='/home/master/Desktop/py/speech_recognition/moje/retrain_logs')
  parser.add_argument('--wanted_words', type=str, default='yes,no,up,down,left,right,on,off,stop,go')
  parser.add_argument('--train_dir', type=str, default='/home/master/Desktop/py/speech_recognition/moje/speech_commands_train')
  parser.add_argument('--save_step_interval', type=int, default=100, help='Save model checkpoint every save_steps.')
  parser.add_argument('--start_checkpoint', type=str, default='', help='If specified, restore this pretrained model before any training.')
  parser.add_argument('--model_architecture', type=str, default='conv', help='What model architecture to use')
  parser.add_argument('--check_nans', type=bool, default=False)
  parser.add_argument('--quantize', type=bool, default=False)
  parser.add_argument('--preprocess', type=str, default='mfcc')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
