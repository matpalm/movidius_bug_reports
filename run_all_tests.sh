#!/usr/bin/env bash
set -ex

./test.sh conv_with_regression                   # PASS

./test.sh conv_with_8_filters_and_padding_valid  # PASS
./test.sh conv_with_8_filters_and_padding_same   # PASS
./test.sh conv_with_1_filter_and_padding_valid   # PASS
./test.sh conv_with_1_filter_and_padding_same    # FAILS sometimes; nan, sometimes inaccurate

#./test.sh deconv_padding_same            # not going to be fixed for a bit

# need to update this to use num_output=1 again
# and not slice since conv_with_1_filter_and_padding_same works
#./test.sh conv_deconv_output_shape_wrong         # FAILS
