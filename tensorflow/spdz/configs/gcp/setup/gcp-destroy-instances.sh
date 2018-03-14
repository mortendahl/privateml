#!/bin/sh

gcloud compute instances delete \
    server0 server1 cryptoproducer inputoutput \
    --quiet