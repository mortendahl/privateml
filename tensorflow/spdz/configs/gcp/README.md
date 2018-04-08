
# Running on GCP

This configuration is for running on the [Google Cloud Platform](https://cloud.google.com/compute/). All steps here assume that the [Cloud SDK](https://cloud.google.com/sdk/) is already installed (for macOS this may e.g. be done using Homebrew: `brew cask install google-cloud-sdk`).

## Setup

Script `gcp-create.sh` provides an easy way to create four GCP instances to be used with this configuration; simply run it from the `setup` directory.

To later remove everything again run `gcp-destroy.sh` from the `setup` directory.

## Starting and Stopping

Right after setup all instances should be stopped, but scripts `start.sh`, `stop.sh`, and `restart.sh` provides easy ways of managing their state. Run them from the `gcp` directory.

`start.sh` will take care of linking the four instances together and starting a TensorFlow server on each in a `screen`.

## Running

The `run.sh` script provides a way to run a program on the four instances. Symlink to it from the `spdz` directory and run it from there, e.g.:

```sh
spdz $ ./run.sh logistic-regression-simple/prediction.py
```

This script will copy the specified file to the input provider, run it, and pull back TensorBoard results to `/tmp/tensorboard`. Run `tensorboard --logdir=/tmp/tensorboard` on the local host to view them.

## Other

Script `ps.sh` can be used to verify that a server is running on each instance. Script `pull.sh` will update the GitHub repository on each instance.