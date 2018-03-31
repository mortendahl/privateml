#!/bin/sh

rm -rf /tmp/tensorboard
rm *.pyc

gcloud compute ssh inputoutput --command='rm -rf /tmp/tensorboard; rm -f ~/*.pyc'
gcloud compute scp $1 inputoutput:./main.py
gcloud compute ssh inputoutput --command='python2 main.py'
gcloud compute scp --recurse inputoutput:/tmp/tensorboard /tmp/tensorboard
