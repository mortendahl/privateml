#!/bin/sh

gcloud compute ssh server0 --command='ps au | egrep python2'
gcloud compute ssh server1 --command='ps au | egrep python2'
gcloud compute ssh cryptoproducer --command='ps au | egrep python2'
gcloud compute ssh inputoutput --command='ps au | egrep python2'