# MTS-DCGAN

**MTS-DCGAN** is a multi-time series anomaly detection framework proposed in our paper ***Robust unsupervised anomaly detection via multi-time scale DCGANs with forgetting mechanism for industrial multivariate time series***.

## Dependencies

TensorFlow > 1.12.0
Keras > 2.2.4

## Usage

In the training and testing, the multi-time series will be encoded as signature matrices first, and then signature matrices will be sent into **MTS-DCGAN** for training or anomay detection. For different datasets, **MTS-DCGAN** has different model parameters and training parameters. Therefore, we introduce the training and testing of **MTS-DCGAN** under different datasets as follows:

### Genesis demonstrator dataset

- Training MTS-DCGAN:
```
python code/Time_Series_Analysis/Anomaly_Detection/core/run_DCGANs.py --model DCGANs --dataset gens --phase train --niter 500 --lr_d 1e-4 --lr_g 4e-4
```

- Testing MTS-DCGAN:
```
python code/Time_Series_Analysis/Anomaly_Detection/core/run_DCGANs.py --model DCGANs --dataset gens --phase test
```

### Shuttle dataset
- Training MTS-DCGAN:
```
python code/Time_Series_Analysis/Anomaly_Detection/core/run_DCGANs.py --model DCGANs --dataset shuttle --phase train --niter 500 --lr_d 1e-4 --lr_g 4e-2
```

- Testing MTS-DCGAN:

```
python code/Time_Series_Analysis/Anomaly_Detection/core/run_DCGANs.py --model DCGANs --dataset shuttle --phase test
```

### Satellite dataset

