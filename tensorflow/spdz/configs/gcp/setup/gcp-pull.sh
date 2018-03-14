#!/bin/sh

gcloud compute ssh server0 < debian-pull.sh
gcloud compute ssh server1 < debian-pull.sh
gcloud compute ssh cryptoproducer < debian-pull.sh
gcloud compute ssh inputoutput < debian-pull.sh
