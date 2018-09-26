from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf


def _next_power_of_two(x):
  
  return 1 if x == 0 else 2**(int(x) - 1).bit_length()


def prepare_model_settings(label_count, sample_rate, clip_duration_ms, window_size_ms, window_stride_ms, feature_bin_count, preprocess):
  desired_samples = int(sample_rate * clip_duration_ms / 1000)
  window_size_samples = int(sample_rate * window_size_ms / 1000)
  window_stride_samples = int(sample_rate * window_stride_ms / 1000)
  length_minus_window = (desired_samples - window_size_samples)
  
  if length_minus_window < 0:
    spectrogram_length = 0
  else:
    spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
  
  if preprocess == 'average':
    fft_bin_count = 1 + (_next_power_of_two(window_size_samples) / 2)
    average_window_width = int(math.floor(fft_bin_count / feature_bin_count))
    fingerprint_width = int(math.ceil(fft_bin_count / average_window_width))
  elif preprocess == 'mfcc':
    average_window_width = -1
    fingerprint_width = feature_bin_count
  else:
    raise ValueError('Unknown preprocess mode "%s" (should be "mfcc" or "average")' % (preprocess))

  fingerprint_size = fingerprint_width * spectrogram_length
  return {
      'desired_samples': desired_samples,
      'window_size_samples': window_size_samples,
      'window_stride_samples': window_stride_samples,
      'spectrogram_length': spectrogram_length,
      'fingerprint_width': fingerprint_width,
      'fingerprint_size': fingerprint_size,
      'label_count': label_count,
      'sample_rate': sample_rate,
      'preprocess': preprocess,
      'average_window_width': average_window_width,
  }


def create_model(fingerprint_input, model_settings, model_architecture, is_training, runtime_settings=None):

  if model_architecture == 'conv':
    return create_conv_model(fingerprint_input, model_settings, is_training)
  elif model_architecture == 'low_latency_conv':
    return create_low_latency_conv_model(fingerprint_input, model_settings, is_training)
  else:
    raise Exception('model_architecture argument not recognized, should be one of "conv", "low_latency_conv"')


def load_variables_from_checkpoint(sess, start_checkpoint):

  saver = tf.train.Saver(tf.global_variables())
  saver.restore(sess, start_checkpoint)


def create_conv_model(fingerprint_input, model_settings, is_training):

  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

  input_frequency_size = model_settings['fingerprint_width']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input, [-1, input_time_size, input_frequency_size, 1])


  first_filter_width = 8
  first_filter_height = 20
  first_filter_count = 64
  first_weights = tf.get_variable(name='first_weights', initializer=tf.truncated_normal_initializer(stddev=0.01),
      shape=[first_filter_height, first_filter_width, 1, first_filter_count])
  first_bias = tf.get_variable(name='first_bias', initializer=tf.zeros_initializer, shape=[first_filter_count])
  first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, 1, 1, 1], 'SAME') + first_bias
  first_relu = tf.nn.relu(first_conv)
  if is_training:
    first_dropout = tf.nn.dropout(first_relu, dropout_prob)
  else:
    first_dropout = first_relu
  max_pool = tf.nn.max_pool(first_dropout, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')


  second_filter_width = 4
  second_filter_height = 10
  second_filter_count = 64
  second_weights = tf.get_variable(name='second_weights', initializer=tf.truncated_normal_initializer(stddev=0.01),
      shape=[second_filter_height, second_filter_width, first_filter_count, second_filter_count])
  second_bias = tf.get_variable(name='second_bias', initializer=tf.zeros_initializer, shape=[second_filter_count])
  second_conv = tf.nn.conv2d(max_pool, second_weights, [1, 1, 1, 1], 'SAME') + second_bias
  second_relu = tf.nn.relu(second_conv)
  if is_training:
    second_dropout = tf.nn.dropout(second_relu, dropout_prob)
  else:
    second_dropout = second_relu


  second_conv_shape = second_dropout.get_shape()
  second_conv_output_width = second_conv_shape[2]
  second_conv_output_height = second_conv_shape[1]
  second_conv_element_count = int(second_conv_output_width * second_conv_output_height * second_filter_count)
  flattened_second_conv = tf.reshape(second_dropout, [-1, second_conv_element_count])
  label_count = model_settings['label_count']
  final_fc_weights = tf.get_variable(name='final_fc_weights', initializer=tf.truncated_normal_initializer(stddev=0.01),
      shape=[second_conv_element_count, label_count])
  final_fc_bias = tf.get_variable(name='final_fc_bias', initializer=tf.zeros_initializer, shape=[label_count])
  final_fc = tf.matmul(flattened_second_conv, final_fc_weights) + final_fc_bias
  if is_training:
    return final_fc, dropout_prob
  else:
    return final_fc


def create_low_latency_conv_model(fingerprint_input, model_settings, is_training):
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

  input_frequency_size = model_settings['fingerprint_width']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input, [-1, input_time_size, input_frequency_size, 1])


  first_filter_width = 8
  first_filter_height = input_time_size
  first_filter_count = 186
  first_filter_stride_x = 1
  first_filter_stride_y = 1
  first_weights = tf.get_variable(name='first_weights', initializer=tf.truncated_normal_initializer(stddev=0.01),
      shape=[first_filter_height, first_filter_width, 1, first_filter_count])
  first_bias = tf.get_variable(name='first_bias', initializer=tf.zeros_initializer, shape=[first_filter_count])
  first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, first_filter_stride_y, first_filter_stride_x, 1], 'VALID') + first_bias
  first_relu = tf.nn.relu(first_conv)
  if is_training:
    first_dropout = tf.nn.dropout(first_relu, dropout_prob)
  else:
    first_dropout = first_relu


  first_conv_output_width = math.floor((input_frequency_size - first_filter_width + first_filter_stride_x) first_filter_stride_x)
  first_conv_output_height = math.floor((input_time_size - first_filter_height + first_filter_stride_y) first_filter_stride_y)
  first_conv_element_count = int(first_conv_output_width * first_conv_output_height * first_filter_count)
  flattened_first_conv = tf.reshape(first_dropout,[-1, first_conv_element_count])
  first_fc_output_channels = 128
  first_fc_weights = tf.get_variable(name='first_fc_weights', initializer=tf.truncated_normal_initializer(stddev=0.01),
      shape=[first_conv_element_count, first_fc_output_channels])
  first_fc_bias = tf.get_variable(name='first_fc_bias', initializer=tf.zeros_initializer, shape=[first_fc_output_channels])
  first_fc = tf.matmul(flattened_first_conv, first_fc_weights) + first_fc_bias


  if is_training:
    second_fc_input = tf.nn.dropout(first_fc, dropout_prob)
  else:
    second_fc_input = first_fc
  second_fc_output_channels = 128
  second_fc_weights = tf.get_variable(name='second_fc_weights', initializer=tf.truncated_normal_initializer(stddev=0.01),
      shape=[first_fc_output_channels, second_fc_output_channels])
  second_fc_bias = tf.get_variable(name='second_fc_bias', initializer=tf.zeros_initializer, shape=[second_fc_output_channels])
  second_fc = tf.matmul(second_fc_input, second_fc_weights) + second_fc_bias


  if is_training:
    final_fc_input = tf.nn.dropout(second_fc, dropout_prob)
  else:
    final_fc_input = second_fc
  label_count = model_settings['label_count']
  final_fc_weights = tf.get_variable(name='final_fc_weights', initializer=tf.truncated_normal_initializer(stddev=0.01),
      shape=[second_fc_output_channels, label_count])
  final_fc_bias = tf.get_variable(name='final_fc_bias', initializer=tf.zeros_initializer, shape=[label_count])
  final_fc = tf.matmul(final_fc_input, final_fc_weights) + final_fc_bias
  if is_training:
    return final_fc, dropout_prob
  else:
    return final_fc
