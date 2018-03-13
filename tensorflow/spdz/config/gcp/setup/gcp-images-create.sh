#!/bin/sh

# create base instance for generating template image
gcloud compute instances create tensorspdz-template \
    --image-project debian-cloud \
    --image-family debian-9 \
    --boot-disk-size 10GB \
    --custom-cpu 4 \
    --custom-memory 15

# install software on it
gcloud compute ssh tensorspdz-template < debian-install.sh

# turn its disk into an image
gcloud compute instances delete tensorspdz-template --keep-disks=boot --quiet
gcloud compute images create tensorspdz-image --source-disk tensorspdz-template
gcloud compute disks delete tensorspdz-template --quiet
