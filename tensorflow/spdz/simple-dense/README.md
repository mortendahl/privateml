# Dense layer example

## Installation

We first need a few Python libraries:

```sh
$ pip install numpy tensorflow tensorboard
```

Git should have automatically created two symlinks in this directory, `config.py` and `tensorspdz.py`, that points to support files, but if not then run

```sh
simple-dense $ ln -s ../config/localhost/config.py config.py
simple-dense $ ln -s ../tensorspdz.py tensorspdz.py
```


## Starting the services

To run this example we need to run the five parties given in `config/localhost/`. These can be launched manually via e.g. `python2 server0.py` but there's also a convenient `start-services.sh` script that will launch all at once. Likewise, `stop-services.sh` may be used to stop all five afterwards.

```sh
localhost $ ./start-services.sh
localhost $ ./stop-services.sh
```

## Running

With all services running simply run

```sh
simple-dense $ python2 main.py
```

## Inspection

Log files from the execution is logging to `/tmp/tensorflow` by default. To visualise these using Tensorboard simply run

```sh
tensorboard --logdir=/tmp/tensorflow
```

and navigate to `http://localhost:6006`.

![Compute time][compute-time]

![Role operations][role-operations]

# Complete GCP setup

In one terminal run the following, starting from your local machine:

```sh
# create new GCP instance
gcloud compute instances create tensorspdz --custom-cpu=1 --custom-memory=6GB
gcloud compute instances start tensorspdz

# connect to it
gcloud compute ssh tensorspdz

# install needed software
sudo apt-get update
sudo apt-get install python python-pip git
pip install numpy tensorflow tensorboard

# clone repo
git clone https://github.com/mortendahl/privateml.git
cd privateml/tensorflow/spdz/config/localhost

# launch services
./start-services.sh
```

In another terminal run the following, starting from your local machine:

```sh
# connect to it
gcloud compute ssh tensorspdz

# run computation
cd privateml/tensorflow/spdz/simple-dense/
python main.py
```

Once done, run the following, starting from your local machine, in any terminal:

```sh
# connect to it
gcloud compute ssh tensorspdz

# stop services
cd privateml/tensorflow/spdz/config/localhost
./stop-services

# disconnect
logoud

# shut down
gcloud compute instances stop tensorspdz
```