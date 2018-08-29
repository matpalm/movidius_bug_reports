#!/usr/bin/env bash
set -ex

./test.sh conv_with_regression            # PASS
./test.sh conv_with_8_filters             # PASS
./test.sh conv_with_6_filters             # FAIL
#./test.sh deconv_padding_same            # not going to be fixed for a bit
./test.sh conv_deconv_output_shape_wrong  # FAILS
