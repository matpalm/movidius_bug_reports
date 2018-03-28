import numpy as np

np.random.seed(123)

def random_img_like_tensor(size):
  return np.random.randint(low=0, high=256, size=size).astype(np.uint8)

# 1024, 768  full
# 512, 384   half
# 256, 192   quarter

#IN_X, IN_Y = 256, 256  # WORKS
#IN_X, IN_Y = 512, 512  # WORKS
IN_X, IN_Y = 64, 64
#IN_X, IN_Y = 256, 192  # FAILS?

#OUT_X, OUT_Y = 1, 1

# tensor to train for 1.0
POS_TENSOR = random_img_like_tensor((IN_X, IN_Y, 3))
POS_LABEL = np.ones((1,))

# tensor to train for 0.0
NEG_TENSOR = random_img_like_tensor((IN_X, IN_Y, 3))
NEG_LABEL = np.zeros((1,))

