#!/usr/bin/env python3

import argparse
import data
import models
import numpy as np
import os
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--eg', type=str, help="which eg we are testing; dictates config & working directory to store graph defs, chpts, compiled models, etc")
opts = parser.parse_args()

# Decide model / examples specific for this test case
pos_tensor, pos_label, neg_tensor, neg_label = data.tensors_for(opts.eg)
imgs, label, loss = models.model_for(opts.eg)

# Train it to madly overfit two specific examples
optimiser = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = optimiser.minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print("pre training; pos_loss", sess.run(loss, feed_dict={imgs: [pos_tensor], label: [pos_label]}),
      "neg_loss", sess.run(loss, feed_dict={imgs: [neg_tensor], label: [neg_label]}))
for _ in range(1000):
  sess.run(train_op, feed_dict={imgs: [pos_tensor], label: [pos_label]})
  sess.run(train_op, feed_dict={imgs: [neg_tensor], label: [neg_label]})
print("post training; pos_loss", sess.run(loss, feed_dict={imgs: [pos_tensor], label: [pos_label]}),
      "neg_loss", sess.run(loss, feed_dict={imgs: [neg_tensor], label: [neg_label]}))

# save model ckpt and export model graph definition
saver = tf.train.Saver()
ckpt_dir = "%s/ckpt" % opts.eg
if not os.path.exists(ckpt_dir):
  os.makedirs(ckpt_dir)
saver.save(sess, "%s/dummy_ckpt" % ckpt_dir)
tf.train.write_graph(sess.graph_def, ".", "%s/graph.pbtxt" % opts.eg)
