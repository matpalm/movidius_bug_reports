import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys

def dump_shape_and_product_of(tag, t):
  shape_product = 1
  for dim in t.get_shape().as_list()[1:]:
    shape_product *= dim
  print("%-10s %-20s #%s" % (tag, t.get_shape(), shape_product), file=sys.stderr)


def model_for(eg):
  if eg == 'conv_with_regression':

    imgs = tf.placeholder(dtype=np.float32, shape=(1, 128, 96, 3), name='imgs')
    dump_shape_and_product_of('imgs', imgs)

    model = slim.conv2d(imgs, num_outputs=32, kernel_size=3, stride=2, scope='e1')
    dump_shape_and_product_of('e1', model)

    model = slim.conv2d(model, num_outputs=64, kernel_size=3, stride=2, scope='e2')
    dump_shape_and_product_of('e2', model)

    # model = slim.conv2d(model, num_outputs=128, kernel_size=3, stride=2, scope='e3')
    # dump_shape_and_product_of('e3', model)

    # model = slim.conv2d(model, num_outputs=128, kernel_size=3, stride=2, scope='e4')
    # dump_shape_and_product_of('e4', model)

    # model = slim.conv2d(model, num_outputs=128, kernel_size=3, stride=2, scope='e5')
    # dump_shape_and_product_of('e5', model)

    model = slim.flatten(model)
    dump_shape_and_product_of('flatten', model)

    # model = slim.fully_connected(inputs=model,
    #                              num_outputs=64,
    #                              scope='h0')
    # dump_shape_and_product_of('h0', model)

#    model = slim.fully_connected(inputs=model,
#                                 num_outputs=64,
#                                 scope='h1')
#    dump_shape_and_product_of('h1', model)

    output = slim.fully_connected(model, num_outputs=1,
                                  activation_fn=None, scope='output')
#    output = tf.nn.sigmoid(logits, name='output')
    dump_shape_and_product_of('output', output)

    label = tf.placeholder(dtype=np.float32, shape=(1, 1), name='label')

    loss = tf.nn.l2_loss(output - label)

    return imgs, label, loss

  elif eg == 'conv_with_6_filters':
    imgs = tf.placeholder(dtype=np.float32, shape=(1, 64, 64, 3), name='imgs')
    dump_shape_and_product_of('imgs', imgs)
    model = slim.conv2d(imgs, num_outputs=6, kernel_size=3, stride=2, scope='e1')
    dump_shape_and_product_of('e1', model)
    model = slim.flatten(model)
    dump_shape_and_product_of('flatten', model)
    logits = slim.fully_connected(model, num_outputs=1, activation_fn=None)
    output = tf.nn.sigmoid(logits, name='output')
    dump_shape_and_product_of('output', output)

    label = tf.placeholder(dtype=np.float32, shape=(1, 1), name='label')

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logits))

    return imgs, label, loss

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
    imgs = tf.placeholder(dtype=np.float32, shape=(1, 128, 128, 3), name='imgs')
    dump_shape_and_product_of('imgs', imgs)

    # conv layer 1 with stride 2 for downsampling
    model = slim.conv2d(imgs, num_outputs=8, kernel_size=3, stride=2, scope='e1')
    dump_shape_and_product_of('e1', model)

    # conv layer 2 with stride 2 for downsampling
    model = slim.conv2d(model, num_outputs=8, kernel_size=3, stride=2, scope='e2')
    dump_shape_and_product_of('e2', model)

    # deconv with stride 2 for upsampling
    # (have to use padding VALID (see deconv_padding_same))
    model = slim.conv2d_transpose(model, num_outputs=8, kernel_size=3, stride=2,
                                  padding='VALID', scope='d1')
    dump_shape_and_product_of('d1', model)

    # use 1x1 conv (with no activation) for logits calculation
    # note: would want num_outputs=1 here, but that fails so instead we
    # slice off first channel of 8.
    logits = slim.conv2d(model, num_outputs=8, kernel_size=1, stride=1,
                         activation_fn=None, scope='logits')
    logits = tf.slice(logits, [0, 0, 0, 0], [1, 65, 65, 1])

    # model output is sigmoid on logits
    output = tf.nn.sigmoid(logits, name='output')
    dump_shape_and_product_of('output', output)

    label = tf.placeholder(dtype=np.float32, shape=(1, 65, 65, 1), name='label')

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logits))

    return imgs, label, loss

  else:
    raise Exception("unknown eg [%s]" % eg)
