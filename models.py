import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys

def dump_shape_and_product_of(tag, t):
  shape_product = 1
  for dim in t.get_shape().as_list()[1:]:
    shape_product *= dim
  print("%-10s %-20s #%s" % (tag, t.get_shape(), shape_product), file=sys.stderr)
  
#def shapes_for(eg):
#  if eg in ['conv_with_8_filters', 'conv_with_6_filters']:
#    return ((1, 64, 64, 3),  # input shape for tf placeholder
#            (1, 1))          # output shape for tf label placeholder
#  else:
#    raise Exception("unknown eg [%s]" % eg)

def model_for(eg):
  if eg == 'conv_with_8_filters':
    imgs = tf.placeholder(dtype=np.float32, shape=(1, 64, 64, 3), name='imgs')
    dump_shape_and_product_of('imgs', imgs)
    model = slim.conv2d(imgs, num_outputs=8, kernel_size=3, stride=2, scope='e1')
    dump_shape_and_product_of('e1', model)
    model = slim.flatten(model)
    dump_shape_and_product_of('flatten', model)
    logits = slim.fully_connected(model, num_outputs=1, activation_fn=None)
    output = tf.nn.sigmoid(logits, name='output')
    dump_shape_and_product_of('output', output)
    label = tf.placeholder(dtype=np.float32, shape=(1, 1), name='label')
    return imgs, logits, label

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
    return imgs, logits, label

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
    return imgs, logits, label

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
    return imgs, logits, label

  else:
    raise Exception("unknown eg [%s]" % eg)
    
