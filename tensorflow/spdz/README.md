
# TensorSpdz

This repository contains an implementation of the SPDZ protocol in TensorFlow as a way of doing private machine learning; see [this blog post](http://mortendahl.github.io/2018/03/01/secure-computation-as-dataflow-programs/) for an introduction.

The `spdz` directory referred to below is the directory where this readme file is located (i.e. `privateml/tensorflow/spdz/`).

## Configuration

The code runs on four processes that first needs to be configured. Many will probably want to start [running locally](./configs/localhost/) but there's also one for [running on GCP](./configs/gcp).

## Examples

To run these examples first make sure to set up one of the configurations above.

### Private logistic regression

This example is divided into two depending on whether training is done publicly or privately.

To do public training and private prediction simply run:

```sh
spdz $ ./run.sh logistic-regression-simple/prediction.py
```

which will first train on a public model, privately distribute the trained parameters to the two servers, and finally make a prediction on a private input so that the two servers know neither the parameters nor the input.

To do both private training and prediction simply run:

```sh
spdz $ ./run.sh logistic-regression-simple/training.py
```

which of course will take longer.
