#!/usr/bin/env python3

import argparse
import tensorflow as tf
import numpy as np
import data
import mvnc.mvncapi as mvnc

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--working-dir', type=str, help="working directory to store graph defs, chpts, compiled models, etc")
opts = parser.parse_args()

# check on host

graph_def = tf.GraphDef()
with open("%s/graph.frozen.pb" % opts.working_dir, "rb") as f:
  graph_def.ParseFromString(f.read())
tf.import_graph_def(graph_def, name=None)

imgs = tf.get_default_graph().get_tensor_by_name('import/imgs:0')
model_output = tf.get_default_graph().get_tensor_by_name('import/output:0')

with tf.Session() as sess:
  host_positive_prediction = sess.run(model_output, feed_dict={imgs: [data.NEG_TENSOR]})[0]
  host_negative_prediction = sess.run(model_output, feed_dict={imgs: [data.POS_TENSOR]})[0]

print("host_positive_prediction", host_positive_prediction)
print("host_negative_prediction", host_negative_prediction)

# check on ncs

devices = mvnc.EnumerateDevices()
if len(devices) == 0:
  raise Exception("no compute stick?")
device = mvnc.Device(devices[0])
device.OpenDevice()

binary_graph = open("%s/graph.mv" % opts.working_dir, 'rb' ).read()
graph = device.AllocateGraph(binary_graph)

def run_on_ncs(input):
  graph.LoadTensor(input.astype(np.float16), '')
  output, _user_object = graph.GetResult()
  return output

ncs_positive_prediction = run_on_ncs(data.NEG_TENSOR)
ncs_negative_prediction = run_on_ncs(data.POS_TENSOR)
print("ncs_positive_prediction", ncs_positive_prediction)
print("ncs_negative_prediction", ncs_negative_prediction)

graph.DeallocateGraph()
device.CloseDevice()

# compare results

pos_close = np.isclose(host_positive_prediction, ncs_positive_prediction, atol=1e-3)
neg_close = np.isclose(host_negative_prediction, ncs_negative_prediction, atol=1e-3)
if pos_close and neg_close:
  print("PASS", opts.working_dir)
else:
  print("FAIL", opts.working_dir)
