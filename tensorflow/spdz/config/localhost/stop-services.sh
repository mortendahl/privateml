#!/bin/sh

kill $(ps aux | egrep -i 'python2? server0.py' | egrep -v 'egrep' | awk '{print $2}') || true
kill $(ps aux | egrep -i 'python2? server1.py' | egrep -v 'egrep' | awk '{print $2}') || true
kill $(ps aux | egrep -i 'python2? cryptoproducer.py' | egrep -v 'egrep' | awk '{print $2}') || true
kill $(ps aux | egrep -i 'python2? inputprovider.py' | egrep -v 'egrep' | awk '{print $2}') || true
kill $(ps aux | egrep -i 'python2? outputreceiver.py' | egrep -v 'egrep' | awk '{print $2}') || true