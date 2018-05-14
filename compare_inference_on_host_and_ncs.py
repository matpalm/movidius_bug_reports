#!/usr/bin/env python3

import argparse
import data
import models
import mvnc.mvncapi as mvnc
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--eg', type=str, help="which eg we are testing; dictates config & working directory to store graph defs, chpts, compiled models, etc")
opts = parser.parse_args()

pos_tensor, pos_label, neg_tensor, neg_label = data.tensors_for(opts.eg)
print("expected positive_prediction", pos_label)
print("expected negativee_prediction", neg_label)

# check host

graph_def = tf.GraphDef()
with open("%s/graph.frozen.pb" % opts.eg, "rb") as f:
  graph_def.ParseFromString(f.read())
tf.import_graph_def(graph_def, name=None)

imgs = tf.get_default_graph().get_tensor_by_name('import/imgs:0')
model_output = tf.get_default_graph().get_tensor_by_name('import/output/BiasAdd:0')

with tf.Session() as sess:
  host_positive_prediction = sess.run(model_output, feed_dict={imgs: [pos_tensor]})[0]
  host_negative_prediction = sess.run(model_output, feed_dict={imgs: [neg_tensor]})[0]
print("host_positive_prediction", host_positive_prediction.shape, host_positive_prediction)
print("host_negative_prediction", host_negative_prediction.shape, host_negative_prediction)

# check on ncs

devices = mvnc.EnumerateDevices()
if len(devices) == 0:
  raise Exception("no compute stick?")
device = mvnc.Device(devices[0])
device.OpenDevice()

binary_graph = open("%s/graph.mv" % opts.eg, 'rb' ).read()
graph = device.AllocateGraph(binary_graph)

def run_on_ncs(input):
  graph.LoadTensor(input.astype(np.float16), '')
  output, _user_object = graph.GetResult()
  return output

ncs_positive_prediction = run_on_ncs(pos_tensor)
ncs_negative_prediction = run_on_ncs(neg_tensor)
print("ncs_positive_prediction", ncs_positive_prediction.shape, ncs_positive_prediction)
print("ncs_negative_prediction", ncs_negative_prediction.shape, ncs_negative_prediction)

graph.DeallocateGraph()
device.CloseDevice()

# compare results

if host_positive_prediction.shape != ncs_positive_prediction.shape:
  raise Exception("shape mismatch between host [%s] and ncs [%s]" % (host_positive_prediction.shape,
                                                                     ncs_positive_prediction.shape))

pos_close = np.isclose(host_positive_prediction, ncs_positive_prediction, atol=1e-3)
neg_close = np.isclose(host_negative_prediction, ncs_negative_prediction, atol=1e-3)
if pos_close and neg_close:
  print("PASS", opts.eg)
else:
  print("FAIL", opts.eg)
