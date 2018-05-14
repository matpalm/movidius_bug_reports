#!/usr/bin/env bash
set -ex

./test.sh conv_with_regression

# TODO: test.sh only supports output node for regression example (output/BiasAdd)
#       would have to refactor this code to (re)support the logistic regression egs

#./test.sh conv_with_8_filters
#./test.sh conv_with_6_filters
#./test.sh deconv_padding_same
#./test.sh conv_deconv_output_shape_wrong
