
# Running

```sh
alias runmain="gcloud compute ssh inputoutput --command='rm -rf /tmp/tensorboard' && gcloud compute scp main.py inputoutput:. && gcloud compute ssh inputoutput --command='python2 main.py' && gcloud compute scp --recurse inputoutput:/tmp/tensorboard /tmp"

runmain
```

# TensorBoard

```sh
gcloud compute scp --recurse inputoutput:/tmp/tensorboard /tmp
```

```sh
gcloud compute ssh inputoutput -- -L 6006:localhost:6006
```



# Installation

```sh
./setup/gcp-create.sh
./setup/gcp-link.py
```

# Starting 

## GCP

```sh
./setup/gcp-start.sh
./setup/gcp-link.py
```

## TensorSpdz

```sh
./start.sh
```

# Stopping

## GCP

```sh
./setup/gcp-stop.sh
```

## TensorSpdz

```sh
./stop.sh
```

# Cleanup

```sh
./setup/gcp-destroy.sh
```

see https://cloud.google.com/solutions/running-distributed-tensorflow-on-compute-engine

# TODO
- upgrade CPU quota