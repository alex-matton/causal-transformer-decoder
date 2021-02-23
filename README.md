# Causal Transformer Decoder

This repository contains the code for the causal transformer decoder, which is
the autoregressive version of the Pytorch TransformerDecoder.

Explanation of what it does can be found here: https://scale.com/blog/pytorch-improvements

### To install the python package directly:

```shell
pip install git+https://github.com/alexmt-scale/causal-transformer-decoder.git
```

### To test the consistency of the implementation:

The Causal Transformer Decoder is supposed to return the same output as the
Pytorch TransformerDecoder when generating sentences, provided the input is the
same.

```shell
python -m pytest
```

### To run benchmarks against Pytorch implementations:
```shell
python -m benchmarks.benchmark_transformers
```

### And against HuggingFace GPT-2 (our implementation should be as fast as theirs):
```shell
python -m benchmarks.gpt_generation_benchmark
```


# License

The source code for the site is licensed under the MIT license, which you can find in the LICENSE file.
