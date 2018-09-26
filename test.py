from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf

# pylint: disable=unused-import
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
# pylint: enable=unused-import

FLAGS = None


def load_graph(filename):
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def load_labels(filename):
  return [line.rstrip() for line in tf.gfile.GFile(filename)]


def run_graph(wav_data, labels, input_layer_name, output_layer_name):
  with tf.Session() as sess:

    #PROSLEDJUJE ZVUK GRAFU
    softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
    predictions, = sess.run(softmax_tensor, {input_layer_name: wav_data})

    #SORTRA PREDVIDJANJA U ODNOSU NA POUZDANOST
    sorted_predictions = predictions.argsort()[-len(labels):][::-1]
    for node_id in sorted_predictions:
      human_string = labels[node_id]
      score = predictions[node_id]
      print('%s (score = %.5f)' % (human_string, score))

    return 0


def test(wav, labels, graph, input_name, output_name):
  if not wav or not tf.gfile.Exists(wav):
    tf.logging.fatal('Audio file does not exist %s', wav)

  if not labels or not tf.gfile.Exists(labels):
    tf.logging.fatal('Labels file does not exist %s', labels)

  if not graph or not tf.gfile.Exists(graph):
    tf.logging.fatal('Graph file does not exist %s', graph)



  labels_list = load_labels(labels)
  print('Labels: ', labels_list)
  print('Testing: ', wav)
  print('')

  #UCITAVA GRAF KOJI SE NALAZI U PODRAZUMEVANOJ SESIJI
  load_graph(graph)

  with open(wav, 'rb') as wav_file:
    wav_data = wav_file.read()

  run_graph(wav_data, labels_list, input_name, output_name)


def main(_):
  test(FLAGS.wav, FLAGS.labels, FLAGS.graph, FLAGS.input_name, FLAGS.output_name)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--wav', type=str, default='', help='Audio file to be identified.')
  parser.add_argument('--graph', type=str, default='', help='Model to use for identification.')
  parser.add_argument('--labels', type=str, default='', help='Path to file containing labels.')
  parser.add_argument('--input_name', type=str, default='wav_data:0', help='Name of WAVE data input node in model.')
  parser.add_argument('--output_name', type=str, default='labels_softmax:0', help='Name of node outputting a prediction in the model.')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)