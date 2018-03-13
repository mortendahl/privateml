#!/bin/sh

HOST_SERVER_0=`gcloud --format='value(networkInterfaces[0].networkIP)' compute instances list server0`
HOST_SERVER_1=`gcloud --format='value(networkInterfaces[0].networkIP)' compute instances list server1`
HOST_CRYPTO_PRODUCER=`gcloud --format='value(networkInterfaces[0].networkIP)' compute instances list cryptoproducer`
HOST_INPUT_PROVIDER=`gcloud --format='value(networkInterfaces[0].networkIP)' compute instances list inputoutput`
HOST_OUTPUT_RECEIVER=`gcloud --format='value(networkInterfaces[0].networkIP)' compute instances list inputoutput`

echo $HOST_SERVER_0
echo $HOST_SERVER_1
echo $HOST_CRYPTO_PRODUCER
echo $HOST_INPUT_PROVIDER
echo $HOST_OUTPUT_RECEIVER