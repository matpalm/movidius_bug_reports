# repro cases for some failing conv models on NCS

included to support [forum bug report](https://ncsforum.movidius.com/discussion/692/incorrect-inference-results-from-a-minimal-tensorflow-model)

see `run_all_tests.sh` for entry point to run each test

currently only `conv_with_regression` with enabled to run; with nothing at all working
i can't be bothered keeping a general piece of code running to reproduce them all.

## conv_with_regression

simple conv net with single dense connection output. trained with two random
examples; one regressing to an output of 10, the other to 5.

```
imgs       (1, 128, 96, 3)      #36864
e1         (1, 64, 48, 32)      #98304
e2         (1, 32, 24, 64)      #49152
flatten    (1, 49152)           #49152
output     (1, 1)               #1
```

all assets are included in repository;
* graph.pbtxt : exported tensorflow graph
* graph.frozen.pb : frozen version of tf graph
* graph.mv : mvncc compiled graph

```
$ tail conv_with_regression/compare_inference_on_host_and_ncs.out
expected positive_prediction [10]
expected negativee_prediction [5]
host_positive_prediction (1,) [ 9.99999523]
host_negative_prediction (1,) [ 5.00000477]
ncs_positive_prediction (1,) [ 9.8203125]
ncs_negative_prediction (1,) [ 4.94921875]
```

## conv_with_8_filters (DISABLED)

example of simple network with conv layer that works.
see tail of `conv_with_8_filters/compare_inference_on_host_and_ncs.out`

```
imgs       (1, 64, 64, 3)       #12288
e1         (1, 32, 32, 8)       #8192
flatten    (1, 8192)            #8192
output     (1, 1)               #1
```

## conv_with_6_filters (DISABLED)

example almost the same as `conv_with_8_filters` but fails.
see tail of `conv_with_6_filters/compare_inference_on_host_and_ncs.out`

```
imgs       (1, 64, 64, 3)       #12288
e1         (1, 32, 32, 6)       #6144
flatten    (1, 6144)            #6144
output     (1, 1)               #1
```

## deconv with padding SAME (DISABLED)

deconv doesn't support padding SAME.
see tail of `deconv_padding_same/mvNCCompile.out`

`[Error 5] Toolkit Error: Stage Details Not Supported: Wrong deconvolution output shape.`

## conv_deconv_output_shape_wrong (DISABLED)

example of conv -> conv -> deconv -> 1x1 conv where output shape is wrong.
note: would want final 1x1 conv to have num_outputs=1 but that doesn't work (see conv_with_6_filters)
so instead use slice to just take first channel of 8.

fails with a incorrect shape coming from NCS model. returns (274625,) instead of expected (65, 65, 1)

```
imgs       (1, 128, 128, 3)     #49152
e1         (1, 64, 64, 8)       #32768
e2         (1, 32, 32, 8)       #8192
d1         (1, 65, 65, 8)       #33800
output     (1, 65, 65, 1)       #4225
```