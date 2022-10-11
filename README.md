## Data Preparation

Download WMT14 En->De data to **ARN/** directory.   

[Link]: https://drive.google.com/file/d/1_dXmqiTKCfg7N41neKzxbgCynts0zXJo/view?usp=sharing

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
