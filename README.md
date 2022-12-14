## Introduction

The Implementation of **《Speeding Up Transformer Decoding via an Attention Refinement Network》** [COLING2022，Long Paper]（Kaixin Wu, Yue Zhang, Bojie Hu, Tong Zhang）

## Data Preparation

Download [WMT14 En->De](https://drive.google.com/file/d/1_dXmqiTKCfg7N41neKzxbgCynts0zXJo/view?usp=sharing) data to **ARN/** directory.

```shell
sh runs/prepare-wmt14-en2de.sh
```

## Model Training

- Transformer Baseline

```shell
sh runs/run-wmt14-en2de-baseline.sh
```

- ARN

```shell
sh runs/run-wmt14-en2de-arn.sh
```
