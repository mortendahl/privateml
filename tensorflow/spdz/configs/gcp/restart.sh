#!/bin/sh

gcloud compute ssh server0 --command='killall python2'
gcloud compute ssh server1 --command='killall python2'
gcloud compute ssh cryptoproducer --command='killall python2'
gcloud compute ssh inputoutput --command='killall python2'

gcloud compute ssh server0 --command='screen -dmS tensorspdz python2 ~/role.py 0'
gcloud compute ssh server1 --command='screen -dmS tensorspdz python2 ~/role.py 1'
gcloud compute ssh cryptoproducer --command='screen -dmS tensorspdz python2 ~/role.py 2'
gcloud compute ssh inputoutput --command='screen -dmS tensorspdz python2 ~/role.py 3'

gcloud compute instances list