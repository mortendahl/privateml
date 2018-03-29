
# TensorSpdz

The `spdz` direction referred to below is the directory where this readme file is located (i.e. `privateml/tensorflow/spdz/`).

[Examples](#examples) including private [prediction]() and [training]() of logistic regression models.


## Examples

To run these examples first make sure to set up one of the [configurations](#configuration) below -- many will probably want to start with the one [running locally](#) but there's also one for [running on GCP](#).

### Private logistic regression

This example is divided into two depending on whether training is done publicly or privately.

To do public training and private prediction simply run:

```sh
spdz $ ./run.sh logistic-regression/prediction.py
```

which will first train on a public model, privately distribute the trained parameters to the two servers, and finally make a prediction on a private input so that the two servers know neither the parameters nor the input.

To do both private training and prediction simply run:

```sh
spdz $ ./run.sh logistic-regression/training.py
```

which of course will take longer.


## Configuration

### Running locally

This is the easiest way to get started since very little setup is required. But since all players are running on the same machine we of course don't have the desired security properties nor accurate performance measures.

To use this configuration we first specify that we wish to it by symlinking `config.py` and `run.sh` to the root directory:

```sh
spdz $ ln -s configs/localhost/config.py config.py
spdz $ ln -s configs/localhost/run.sh run.sh
```

and simply start the five local processes:

```sh
spdz $ cd configs/localhost/
localhost $ ./start.sh
```

This will block the current terminal, so to actually execute any of the examples we need another terminal.

A script is also provided to stop these processes again (but be a bit careful as it's currently not doing so in a particularly nice way):

```sh
spdz $ cd configs/localhost/
localhost $ ./stop.sh
```

At this point it's safe to delete the symlink again in case you e.g. wish to experiment with another configuration:

```sh
spdz $ rm config.py \
       rm run.sh
```

### Running on GCP

<em>(coming soon..)</em>

```sh
spdz $ cd configs/gcp/