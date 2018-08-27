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

  elif eg in ['conv_with_8_filters', 'conv_with_6_filters', 'deconv_padding_same']:
    np.random.seed(123)
    return (random_img_like_tensor((64, 64, 3)),  # pos example
            np.ones((1,)),                        # pos label
            random_img_like_tensor((64, 64, 3)),  # neg example
            np.zeros((1,)))                       # neg label

  elif eg == 'conv_deconv_output_shape_wrong':
    np.random.seed(123)
    return (random_img_like_tensor((128, 128, 3)),  # pos example
            np.ones((65, 65, 1)),                   # pos label
            random_img_like_tensor((128, 128, 3)),  # neg example
            np.zeros((65, 65, 1)))                  # neg label

  else:
    raise Exception("unknown eg [%s]" % eg)
