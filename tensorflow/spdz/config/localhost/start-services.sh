#!/bin/sh

python2 server0.py &
python2 server1.py &
python2 cryptoproducer.py &
python2 inputprovider.py &
python2 outputreceiver.py