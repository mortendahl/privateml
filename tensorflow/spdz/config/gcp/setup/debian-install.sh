#!/bin/sh

# install OS packages
sudo apt-get update
sudo apt-get -y upgrade
sudo apt-get install -y python-pip git

# install python packages
sudo pip install numpy tensorflow tensorboard

# repo
cd ~
git clone https://github.com/mortendahl/privateml.git
