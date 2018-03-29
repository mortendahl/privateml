#!/bin/sh

gcloud compute instances stop \
    server0 server1 cryptoproducer inputoutput

gcloud compute instances list