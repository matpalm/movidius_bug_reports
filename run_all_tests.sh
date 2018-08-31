#!/usr/bin/env bash
set -ex

./test.sh conv_with_regression                   # PASS

./test.sh conv_with_8_filters_and_padding_valid  # PASS
./test.sh conv_with_8_filters_and_padding_same   # PASS
./test.sh conv_with_1_filter_and_padding_valid   # PASS
./test.sh conv_with_1_filter_and_padding_same    # FAILS sometimes; nan, sometimes inaccurate

#./test.sh deconv_padding_same            # not going to be fixed for a bit

# example of simple unet conv -> deconv architecture
./test.sh conv_deconv_output_shape_wrong         # PASSES, but requires a hack on NCS output
