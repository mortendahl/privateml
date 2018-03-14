#!/bin/sh

# clean up from creation, in case it got aborted
gcloud compute instances delete tensorspdz-template --quiet
gcloud compute disks delete tensorspdz-template --quiet

# actual purpose
gcloud compute images delete tensorspdz-image --quiet