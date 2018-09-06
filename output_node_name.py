#!/usr/bin/env python3
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--eg', type=str, help="which eg we are testing; dictates config & working directory to store graph defs, chpts, compiled models, etc")
opts = parser.parse_args()

if opts.eg == 'conv_with_regression':
  print("output/BiasAdd")
elif opts.eg == 'conv_deconv_output_shape_wrong':
  # HACK: sigmoid output on tensor > (63,63,1) gives read error
  print("logits/BiasAdd")
else:
  print("output")
