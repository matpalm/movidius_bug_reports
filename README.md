# repro cases for some failing conv models on NCS

see `run_all_tests.sh` for entry point to run each test

## conv_with_8_filters

example of simple network with conv layer that works.
see tail of `conv_with_8_filters/compare_inference_on_host_and_ncs.out`

```
imgs       (1, 64, 64, 3)       #12288
e1         (1, 32, 32, 8)       #8192
flatten    (1, 8192)            #8192
output     (1, 1)               #1
```

## conv_with_6_filters

example almost the same as `conv_with_8_filters` but fails.
see tail of `conv_with_6_filters/compare_inference_on_host_and_ncs.out`

```
imgs       (1, 64, 64, 3)       #12288
e1         (1, 32, 32, 6)       #6144
flatten    (1, 6144)            #6144
output     (1, 1)               #1
```

## deconv with padding SAME

deconv doesn't support padding SAME.
see tail of `deconv_padding_same/mvNCCompile.out`

`[Error 5] Toolkit Error: Stage Details Not Supported: Wrong deconvolution output shape.`

## conv_deconv_output_shape_wrong

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