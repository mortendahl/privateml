#!/bin/sh

gcloud compute ssh server0 --command='cd privateml && git reset --hard && git pull'
gcloud compute ssh server1 --command='cd privateml && git reset --hard && git pull'
gcloud compute ssh cryptoproducer --command='cd privateml && git reset --hard && git pull'
gcloud compute ssh inputoutput --command='cd privateml && git reset --hard && git pull'
