#!/usr/bin/env python3

import argparse
import data
import models
import mvnc.mvncapi as mvnc
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--eg', type=str, help="which eg we are testing; dictates config & working directory to store graph defs, chpts, compiled models, etc")
parser.add_argument('--output-node-name', type=str, help="model output node name")
opts = parser.parse_args()

pos_tensor, pos_label, neg_tensor, neg_label = data.tensors_for(opts.eg)
print("expected positive_prediction", pos_label.flatten()[:10])
print("expected negativee_prediction", neg_label.flatten()[:10])

# check host

graph_def = tf.GraphDef()
with open("%s/graph.frozen.pb" % opts.eg, "rb") as f:
  graph_def.ParseFromString(f.read())
tf.import_graph_def(graph_def, name=None)

imgs = tf.get_default_graph().get_tensor_by_name('import/imgs:0')
model_output = tf.get_default_graph().get_tensor_by_name("import/%s:0" % opts.output_node_name)

with tf.Session() as sess:
  host_positive_prediction = sess.run(model_output, feed_dict={imgs: [pos_tensor]})[0]
  host_negative_prediction = sess.run(model_output, feed_dict={imgs: [neg_tensor]})[0]
print("host_positive_prediction", host_positive_prediction.shape, host_positive_prediction.flatten()[:10])
print("host_negative_prediction", host_negative_prediction.shape, host_negative_prediction.flatten()[:10])

# check on ncs

devices = mvnc.enumerate_devices()
if len(devices) == 0:
  raise Exception("no compute stick?")
device = mvnc.Device(devices[0])
device.open()

binary_graph = open("%s/graph.mv" % opts.eg, 'rb' ).read()
graph = mvnc.Graph('g')
input_fifo, output_fifo = graph.allocate_with_fifos(device, binary_graph)

def run_on_ncs(input):
  graph.queue_inference_with_fifo_elem(input_fifo, output_fifo,
                                       np.float32(input), None)
  output, _user_object = output_fifo.read_elem()
  return output

ncs_positive_prediction = run_on_ncs(pos_tensor)
ncs_negative_prediction = run_on_ncs(neg_tensor)
print("ncs_positive_prediction", ncs_positive_prediction.shape, ncs_positive_prediction.flatten()[:10])
print("ncs_negative_prediction", ncs_negative_prediction.shape, ncs_negative_prediction.flatten()[:10])

if opts.eg == 'conv_deconv_output_shape_wrong':
  N = 63
  ncs_positive_prediction = ncs_positive_prediction[:N*N].reshape((N,N,1))
  ncs_negative_prediction = ncs_negative_prediction[:N*N].reshape((N,N,1))

input_fifo.destroy()
output_fifo.destroy()
graph.destroy()
device.close()
device.destroy()

# compare results

if host_positive_prediction.shape != ncs_positive_prediction.shape:
  raise Exception("shape mismatch between host [%s] and ncs [%s]" % (host_positive_prediction.shape,
                                                                     ncs_positive_prediction.shape))

pos_close = np.all(np.isclose(host_positive_prediction, ncs_positive_prediction, atol=0.2))
neg_close = np.all(np.isclose(host_negative_prediction, ncs_negative_prediction, atol=0.2))
if pos_close and neg_close:
  print("PASS", opts.eg)
else:
  print("FAIL", opts.eg)
