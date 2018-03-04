#!/bin/sh

kill $(ps | egrep -i 'python server0.py' | egrep -v 'egrep' | awk '{print $1}') || true
kill $(ps | egrep -i 'python server1.py' | egrep -v 'egrep' | awk '{print $1}') || true
kill $(ps | egrep -i 'python cryptoproducer.py' | egrep -v 'egrep' | awk '{print $1}') || true
kill $(ps | egrep -i 'python inputprovider.py' | egrep -v 'egrep' | awk '{print $1}') || true
kill $(ps | egrep -i 'python outputreceiver.py' | egrep -v 'egrep' | awk '{print $1}') || true