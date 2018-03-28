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
imgs, logits, label = models.model_for(opts.eg)

# Train it to madly overfit two specific examples
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logits))
optimiser = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimiser.minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for _ in range(100):
  sess.run(train_op, feed_dict={imgs: [pos_tensor], label: [pos_label]})
  sess.run(train_op, feed_dict={imgs: [neg_tensor], label: [neg_label]})

# save model ckpt and export model graph definition
saver = tf.train.Saver()
ckpt_dir = "%s/ckpt" % opts.eg
if not os.path.exists(ckpt_dir):
  os.makedirs(ckpt_dir)
saver.save(sess, "%s/dummy_ckpt" % ckpt_dir)
tf.train.write_graph(sess.graph_def, ".", "%s/graph.pbtxt" % opts.eg)



