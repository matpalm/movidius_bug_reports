import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys

def dump_shape_and_product_of(tag, t):
  shape_product = 1
  for dim in t.get_shape().as_list()[1:]:
    shape_product *= dim
  print("%-10s %-20s #%s" % (tag, t.get_shape(), shape_product), file=sys.stderr)

def conv_with_n_filters(n, padding):
    imgs = tf.placeholder(dtype=np.float32, shape=(1, 64, 64, 3), name='imgs')
    dump_shape_and_product_of('imgs', imgs)
    model = slim.conv2d(imgs, num_outputs=n, kernel_size=3, stride=2, padding=padding, scope='e1')
    dump_shape_and_product_of('e1', model)
    model = slim.flatten(model)
    dump_shape_and_product_of('flatten', model)
    logits = slim.fully_connected(model, num_outputs=1, activation_fn=None)
    output = tf.nn.sigmoid(logits, name='output')
    dump_shape_and_product_of('output', output)
    label = tf.placeholder(dtype=np.float32, shape=(1, 1), name='label')
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logits))
    return imgs, label, loss


def model_for(eg):

  if eg == 'conv_with_regression':
    # see this model fail if the spatial input is large enough input size that
    # the flatten goes above some size.
    imgs = tf.placeholder(dtype=np.float32, shape=(1, 128, 96, 3), name='imgs')
    dump_shape_and_product_of('imgs', imgs)
    model = slim.conv2d(imgs, num_outputs=32, kernel_size=3, stride=2, scope='e1')
    dump_shape_and_product_of('e1', model)
    model = slim.conv2d(model, num_outputs=64, kernel_size=3, stride=2, scope='e2')
    dump_shape_and_product_of('e2', model)
    model = slim.flatten(model)
    dump_shape_and_product_of('flatten', model)
    output = slim.fully_connected(model, num_outputs=1,
                                  activation_fn=None, scope='output')

    #TODO: what is the right thing to do here to introduce a no op operation
    #      whose only purpose is to ensure there is a node here named just
    #      'output' as opposed to 'output/BiasAdd' to make this model consistent
    #      with the following examples...

    dump_shape_and_product_of('output', output)
    label = tf.placeholder(dtype=np.float32, shape=(1, 1), name='label')
    loss = tf.nn.l2_loss(output - label)
    return imgs, label, loss

  elif eg == 'conv_with_8_filters_and_padding_valid':
    return conv_with_n_filters(n=8, padding='VALID')

  elif eg == 'conv_with_8_filters_and_padding_same':
    return conv_with_n_filters(n=8, padding='SAME')

  elif eg == 'conv_with_1_filter_and_padding_valid':
    return conv_with_n_filters(n=1, padding='VALID')

  elif eg == 'conv_with_1_filter_and_padding_same':
    return conv_with_n_filters(n=1, padding='SAME')

  elif eg == 'deconv_padding_same':
    imgs = tf.placeholder(dtype=np.float32, shape=(1, 64, 64, 3), name='imgs')
    dump_shape_and_product_of('imgs', imgs)
    model = slim.conv2d_transpose(imgs, num_outputs=6, kernel_size=3, stride=2,
                                  padding='SAME', scope='d1')
    dump_shape_and_product_of('d1', model)
    model = slim.flatten(model)
    dump_shape_and_product_of('flatten', model)
    logits = slim.fully_connected(model, num_outputs=1, activation_fn=None)
    output = tf.nn.sigmoid(logits, name='output')
    dump_shape_and_product_of('output', output)

    label = tf.placeholder(dtype=np.float32, shape=(1, 1), name='label')

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logits))

    return imgs, label, loss

  elif eg == 'conv_deconv_output_shape_wrong':

    # if i can't get an output of 1 channel working i'll have to do 8 filter
    # and then slice off the first (see conv_with_1_filter) trouble is the
    # shape of the slicing is wrong in the ncs inference case... note: this might
    # be related to the slice, which i only include because conv with 1 filter fails

    imgs = tf.placeholder(dtype=np.float32, shape=(1, 64, 64, 3), name='imgs')
    dump_shape_and_product_of('imgs', imgs)

    # conv layer 1 with stride 2 for downsampling
    model = slim.conv2d(imgs, num_outputs=8, kernel_size=3, stride=2, padding='VALID', scope='e1')
    dump_shape_and_product_of('e1', model)

    # conv layer 2 with stride 2 for downsampling
    model = slim.conv2d(model, num_outputs=16, kernel_size=3, stride=2, padding='VALID', scope='e2')
    dump_shape_and_product_of('e2', model)

    # deconv with stride 2 for upsampling
    model = slim.conv2d_transpose(model, num_outputs=8, kernel_size=3, stride=2,
                                  padding='VALID', scope='d1')
    dump_shape_and_product_of('d1', model)

    # use 1x1 conv with single kernel, with no activation, for logits calculation
    logits = slim.conv2d(model, num_outputs=1, kernel_size=1, stride=1,
                         padding='VALID',  activation_fn=None, scope='logits')

    # model output is sigmoid on logits
    output = tf.nn.sigmoid(logits, name='output')
    dump_shape_and_product_of('output', output)

    label = tf.placeholder(dtype=np.float32, shape=(1, 31, 31, 1), name='label')

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logits))

    return imgs, label, loss

  else:
    raise Exception("unknown eg [%s]" % eg)
