#!/bin/sh

# repo
cd ~
git clone https://github.com/mortendahl/privateml.git

# pull
cd privateml
git reset --hard origin HEAD
git pull
