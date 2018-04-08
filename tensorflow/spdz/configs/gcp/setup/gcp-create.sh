#!/bin/sh

sh ./gcp-create-images.sh
sh ./gcp-create-instances.sh

gcloud compute instances stop \
    server0 server1 cryptoproducer inputoutput
