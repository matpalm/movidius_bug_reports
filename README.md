# repro cases for some failing conv models on NCS

included to support [forum bug report](https://ncsforum.movidius.com/discussion/692/incorrect-inference-results-from-a-minimal-tensorflow-model)

see `run_all_tests.sh` for entry point to run each test

for each test a bunch of the assets are checked in, see directory corresponding to the test

includes ...

* graph.pbtxt : exported tensorflow graph
* graph.frozen.pb : frozen version of tf graph
* graph.mv : mvncc compiled graph

tl;dr of all tests; use `padding='VALID'` over `padding='SAME'`

## conv_with_regression  (PASSING)

simple conv net with single dense connection output. trained with two random
examples; one regressing to an output of 10, the other to 5.

this test works currently but is included because if we increase the img input
size it makes the flatten output too big and we get corruptiong and the test fails.
i actually don't care about this specific case, but include this one as a form
of smoke test.

```
imgs       (1, 128, 96, 3)      #36864
e1         (1, 64, 48, 32)      #98304
e2         (1, 32, 24, 64)      #49152
flatten    (1, 49152)           #49152
output     (1, 1)               #1
```

```
$ tail conv_with_regression/compare_inference_on_host_and_ncs.out
expected positive_prediction [10]
expected negativee_prediction [5]
host_positive_prediction (1,) [ 9.99999523]
host_negative_prediction (1,) [ 5.00000477]
ncs_positive_prediction (1,) [ 9.8203125]
ncs_negative_prediction (1,) [ 4.94921875]
```

## conv_with_8_filters_and_padding_valid  (PASSING)

example of simple network with conv layer that works.

```
imgs       (1, 64, 64, 3)       #12288
e1         (1, 31, 31, 8)       #7688
flatten    (1, 7688)            #7688
output     (1, 1)               #1
```

```
host_positive_prediction (1,) [ 0.95524758]
host_negative_prediction (1,) [ 0.04408115]
ncs_positive_prediction (1,) [ 0.95507812]
ncs_negative_prediction (1,) [ 0.04492188]
```

## conv_with_8_filters_and_padding_same  (PASSING)

```
imgs       (1, 64, 64, 3)       #12288
e1         (1, 32, 32, 8)       #8192
flatten    (1, 8192)            #8192
output     (1, 1)               #1
```

```
host_positive_prediction (1,) [ 0.96249026]
host_negative_prediction (1,) [ 0.03636319]
ncs_positive_prediction (1,) [ 0.96240234]
ncs_negative_prediction (1,) [ 0.03619385]
```

## conv_with_1_filter_and_padding_valid  (PASSING)

a 1 filter output is the target use case for
[beeNN](https://github.com/matpalm/bnn) where the networks final
output is a 1 channel bitmap. with `padding='VALID'` this works

```
imgs       (1, 64, 64, 3)       #12288
e1         (1, 31, 31, 1)       #961
flatten    (1, 961)             #961
output     (1, 1)               #1
```

```
host_positive_prediction (1,) [ 0.88046074]
host_negative_prediction (1,) [ 0.11214196]
ncs_positive_prediction (1,) [ 0.88037109]
ncs_negative_prediction (1,) [ 0.11230469]
```

## conv_with_1_filter_and_padding_same  (FAILING)

... but same network with `padding='SAME'` fails

```
imgs       (1, 64, 64, 3)       #12288
e1         (1, 32, 32, 1)       #1024
flatten    (1, 1024)            #1024
output     (1, 1)               #1
```

```
host_positive_prediction (1,) [ 0.90099496]
host_negative_prediction (1,) [ 0.10527134]
ncs_positive_prediction (1,) [ 0.65576172]
ncs_negative_prediction (1,) [ 0.36425781]
```

## deconv with padding SAME (DISABLED)

deconv doesn't support padding SAME. this is annoying for BNN but i can work around it.

see tail of `deconv_padding_same/mvNCCompile.out`

`[Error 5] Toolkit Error: Stage Details Not Supported: Wrong deconvolution output shape.`

## conv_deconv_output_shape_wrong (DISABLED)

TODO: redo this test, can get `num_channels=1` to work with `padding=VALID`

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