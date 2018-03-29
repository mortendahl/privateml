#!/bin/sh

gcloud compute ssh server0 --command='./pull.sh'
gcloud compute ssh server1 --command='./pull.sh'
gcloud compute ssh cryptoproducer --command='./pull.sh'
gcloud compute ssh inputoutput --command='./pull.sh'
