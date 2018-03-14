#!/bin/sh

# install OS packages
sudo apt-get update
sudo apt-get -y upgrade
sudo apt-get install -y python-pip 
sudo apt-get install -y git
sudo apt-get install -y psmisc # for killall

# install python packages
sudo pip install numpy tensorflow tensorboard

# repo
cd ~
git clone https://github.com/mortendahl/privateml.git

# links
ln -s ~/privateml/tensorflow/spdz/tensorspdz.py ~/tensorspdz.py
ln -s ~/privateml/tensorflow/spdz/config/gcp/config.py ~/config.py
ln -s ~/privateml/tensorflow/spdz/config/gcp/role.py ~/role.py
