#!/usr/bin/env python2

import json
import subprocess

BASE_PORT = 4440
HOSTS_FILE = '/tmp/hosts.json'
INSTANCE_NAMES = [
    'server0',
    'server1',
    'cryptoproducer',
    'inputoutput'
]
CMD_INTERNAL_HOST = "gcloud --format='value(networkInterfaces[0].networkIP)' compute instances list {instance_name}"
CMD_HOSTS_SCP = "gcloud compute scp {hosts_file} {instance_name}:{hosts_file}"

HOSTS = [
    subprocess.check_output(CMD_INTERNAL_HOST.format(instance_name=instance_name), shell=True).strip()
    for instance_name in INSTANCE_NAMES
]

HOSTS = [
    host + ':' + str(BASE_PORT + index)
    for index, host in enumerate(HOSTS)
]

print HOSTS

with open(HOSTS_FILE, 'w') as f:
    json.dump(HOSTS, f)

for instance_name in INSTANCE_NAMES:
    subprocess.call(CMD_HOSTS_SCP.format(hosts_file=HOSTS_FILE, instance_name=instance_name), shell=True)