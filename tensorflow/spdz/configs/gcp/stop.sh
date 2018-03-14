#!/bin/sh

gcloud compute ssh server0 --command='killall python2'
gcloud compute ssh server1 --command='killall python2'
gcloud compute ssh cryptoproducer --command='killall python2'
gcloud compute ssh inputoutput --command='killall python2'
