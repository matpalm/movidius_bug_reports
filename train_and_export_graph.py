#!/usr/bin/env python3

import argparse
import data
import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--working-dir', type=str, help="working directory to store graph defs, chpts, compiled models, etc")
opts = parser.parse_args()

def dump_shape_and_product_of(tag, t):
  shape_product = 1
  for dim in t.get_shape().as_list()[1:]:
    shape_product *= dim
  print("%-10s %-20s #%s" % (tag, t.get_shape(), shape_product), file=sys.stderr)

# Define a trivial model; some convolutions, dense connections, and a binary prediction
imgs_shape = (1, data.IN_X, data.IN_Y, 3)
imgs = tf.placeholder(dtype=np.float32, shape=imgs_shape, name='imgs')
dump_shape_and_product_of('imgs', imgs)

model = slim.conv2d(imgs, num_outputs=8, kernel_size=3, stride=2, scope='e1')
dump_shape_and_product_of('e1', model)

model = slim.flatten(model)
dump_shape_and_product_of('flatten', model)

logits = slim.fully_connected(model, num_outputs=1, activation_fn=None)
output = tf.nn.sigmoid(logits, name='output')
dump_shape_and_product_of('output', output)

# Train it to madly overfit two specific known examples (described in data.py)
label = tf.placeholder(dtype=np.float32, shape=(1, 1), name='label')
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logits))
optimiser = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train_op = optimiser.minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# to avoid tf.convert_image_dtype in graph (which isn't supported by
# ncs) we do explicit conversion here
def convert_image_dtype(uint8_img):
  # convert a (0, 255) uint8 array to (-1.0,1.0) float32
  float32_img = uint8_img.astype(np.float32)  # (0.0, 255.0)
  float32_img /= 128                          # (0.0, 2.0)
  return float32_img - 1.0                    # (-1.0, 1.0)

#def convert_label_dtype(uint8_img):
#  float32_img = uint8_img.astype(np.float32)  # (0.0, 255.0)
#  float32_img /= 255                          # (0.0, 1.0)
#  return float32_img

pos_img = [convert_image_dtype(data.POS_TENSOR)]
pos_label = [data.POS_LABEL]
neg_img = [convert_image_dtype(data.NEG_TENSOR)]
neg_label = [data.NEG_LABEL]

for _ in range(100):
  sess.run(train_op, feed_dict={imgs: pos_img, label: pos_label})
  sess.run(train_op, feed_dict={imgs: neg_img, label: neg_label})

# save model ckpt and export model graph definition
saver = tf.train.Saver()
ckpt_dir = "%s/ckpt" % opts.working_dir
if not os.path.exists(ckpt_dir):
  os.makedirs(ckpt_dir)
saver.save(sess, "%s/dummy_ckpt" % ckpt_dir)
tf.train.write_graph(sess.graph_def, ".", "%s/graph.pbtxt" % opts.working_dir)



