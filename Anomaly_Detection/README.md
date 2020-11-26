# MTS-DCGAN

**MTS-DCGAN** is a multi-time series anomaly detection framework proposed in our paper ***Robust unsupervised anomaly detection via multi-time scale DCGANs with forgetting mechanism for industrial multivariate time series***.

## Dependencies

TensorFlow > 1.12.0

Keras > 2.2.4

## Usage

In the training and testing, the multi-time series will be encoded as signature matrices first, and then signature matrices will be sent into **MTS-DCGAN** for training or anomay detection. For different datasets, **MTS-DCGAN** has different model parameters and training parameters. Therefore, we introduce the training and testing of **MTS-DCGAN** under different datasets as follows:

### Genesis demonstrator dataset

- *Training MTS-DCGAN:*
```
python Anomaly_Detection/core/run_DCGANs.py --model DCGANs --dataset gens --phase train --niter 500 --lr_d 1e-4 --lr_g 4e-4
```

- *Testing MTS-DCGAN:*
```
python Anomaly_Detection/core/run_DCGANs.py --model DCGANs --dataset gens --phase test
```

### Shuttle dataset
- *Training MTS-DCGAN:*
```
python Anomaly_Detection/core/run_DCGANs.py --model DCGANs --dataset shuttle --phase train --niter 500 --lr_d 1e-4 --lr_g 4e-2
```

- *Testing MTS-DCGAN:*

```
python Anomaly_Detection/core/run_DCGANs.py --model DCGANs --dataset shuttle --phase test
```

### Satellite dataset
- *Training MTS-DCGAN:*
```
python Anomaly_Detection/core/run_DCGANs.py --model DCGANs --dataset satellite --phase train --niter 500 --lr_d 1e-4 --lr_g 1e-4
```

- *Testing MTS-DCGAN:*

```
python Anomaly_Detection/core/run_DCGANs.py --model DCGANs --dataset satellite --phase test
```

### Gamma dataset
- *Training MTS-DCGAN:*
```
python Anomaly_Detection/core/run_DCGANs.py --model DCGANs --dataset gamaray --phase train --niter 500 --lr_d 1e-4 --lr_g 2e-4
```

- *Testing MTS-DCGAN:*

```
python Anomaly_Detection/core/run_DCGANs.py --model DCGANs --dataset gamaray --phase test
```

## Citing MTS-DCGAN:

If you use this repository or would like to refer the paper, please use the following BibTeX entry:

```
@article{LIANG2020,
title = "Robust unsupervised anomaly detection via multi-time scale DCGANs with forgetting mechanism for industrial multivariate time series",
journal = "Neurocomputing",
year = "2020",
issn = "0925-2312",
doi = "https://doi.org/10.1016/j.neucom.2020.10.084",
url = "http://www.sciencedirect.com/science/article/pii/S0925231220316970",
author = "Haoran Liang and Lei Song and Jianxing Wang and Lili Guo and Xuzhi Li and Ji Liang",
keywords = "Multivariate time series, Unsupervised anomaly detection, Multi-time scale, Deep convolutional generative adversarial networks, Threshold setting strategy"
}
```
