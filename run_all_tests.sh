#!/usr/bin/env bash
set -ex

./test.sh conv_with_8_filters
./test.sh conv_with_6_filters
./test.sh deconv_padding_same
./test.sh conv_deconv_output_shape_wrong

