#!/usr/bin/env bash
set -x

rm -rf $1
mkdir $1

set -e

# this is kinda clumsy but...
OUTPUT_NODE_NAME=`./output_node_name.py --eg $1`

./train_and_export_graph.py --eg $1 2>&1 | tee $1/train_and_export_graph.out

python3 -m tensorflow.python.tools.freeze_graph \
 --clear_devices \
 --input_graph $1/graph.pbtxt \
 --input_checkpoint $1/ckpt/dummy_ckpt \
 --output_node_names "$OUTPUT_NODE_NAME" \
 --output_graph $1/graph.frozen.pb 2>&1 | tee $1/freeze_graph.out

mvNCCompile $1/ckpt/dummy_ckpt.meta -s 12 -in imgs -on $OUTPUT_NODE_NAME -o $1/graph.mv 2>&1 | tee $1/mvNCCompile.out

./compare_inference_on_host_and_ncs.py --eg $1 --output-node-name $OUTPUT_NODE_NAME 2>&1 | tee $1/compare_inference_on_host_and_ncs.out
