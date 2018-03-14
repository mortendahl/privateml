#!/bin/sh

gcloud compute instances start \
    server0 server1 cryptoproducer inputoutput

gcloud compute instances list