#!/bin/sh

gcloud compute instances create \
    server0 server1 cryptoproducer inputoutput \
    --image tensorspdz-image \
    --custom-cpu 2 \
    --custom-memory 10
