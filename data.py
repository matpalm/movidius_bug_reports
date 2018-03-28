import numpy as np

def random_img_like_tensor(size):
  return np.random.randint(low=0, high=256, size=size).astype(np.uint8)

def tensors_for(eg):
  if eg in ['conv_with_8_filters', 'conv_with_6_filters']:
    np.random.seed(123)
    return (random_img_like_tensor((64, 64, 3)),  # pos example
            np.ones((1,)),                        # pos label
            random_img_like_tensor((64, 64, 3)),  # neg example
            np.zeros((1,)))                       # neg label
  else:
    raise Exception("unknown eg [%s]" % eg)


