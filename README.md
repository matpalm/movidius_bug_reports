# repro cases for some failing conv models on NCS

see `run_all_tests.sh` for entry point to run each test

## conv_with_8_filters

example of simple network with conv layer that works.
see tail of `conv_with_8_filters/compare_inference_on_host_and_ncs.out`

## conv_with_6_filters

example almost the same as `conv_with_8_filters` but fails.
see tail of `conv_with_6_filters/compare_inference_on_host_and_ncs.out`

## deconv with padding SAME

deconv doesn't support padding SAME.
see tail of `deconv_padding_same/mvNCCompile.out`

`[Error 5] Toolkit Error: Stage Details Not Supported: Wrong deconvolution output shape.`