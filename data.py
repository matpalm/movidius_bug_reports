import numpy as np

def random_img_like_tensor(size):
  return np.random.random(size=size)

def tensors_for(eg):
  if eg in ['conv_with_regression']:
    np.random.seed(123)
    return (random_img_like_tensor((128, 96, 3)),  # pos example
            np.array([10]),                        # pos label
            random_img_like_tensor((128, 96, 3)),  # neg example
            np.array([5]))                         # neg label

  elif eg in ['conv_with_8_filters_and_padding_valid',
              'conv_with_8_filters_and_padding_same',
              'conv_with_1_filter_and_padding_valid',
              'conv_with_1_filter_and_padding_same',
              'deconv_padding_same']:
    np.random.seed(123)
    return (random_img_like_tensor((64, 64, 3)),  # pos example
            np.ones((1,)),                        # pos label
            random_img_like_tensor((64, 64, 3)),  # neg example
            np.zeros((1,)))                       # neg label

  elif eg == 'conv_deconv_output_shape_wrong':
    np.random.seed(123)
    I, O = 512, 127
    return (random_img_like_tensor((I, I, 3)),  # pos example
            np.ones((O, O, 1)),                 # pos label
            random_img_like_tensor((I, I, 3)),  # neg example
            np.zeros((O, O, 1)))                # neg label

  else:
    raise Exception("unknown eg [%s]" % eg)
