#!/bin/sh

echo 'export LANGUAGE=en_US.UTF-8' >> .bashrc
echo 'export LANG=en_US.UTF-8' >> .bashrc
echo 'export LC_ALL=en_US.UTF-8' >> .bashrc

# install OS packages
sudo apt-get update
sudo apt-get -y upgrade
sudo apt-get install -y python-pip 
sudo apt-get install -y git
sudo apt-get install -y psmisc # for killall
sudo apt-get install -y tcptrack iftop # for network monitoring

# install python packages
sudo pip install numpy tensorflow tensorboard

# repo
cd ~
git clone https://github.com/mortendahl/privateml.git

# links
ln -s ~/privateml/tensorflow/spdz/tensorspdz.py ~/tensorspdz.py
ln -s ~/privateml/tensorflow/spdz/configs/gcp/server/config.py ~/config.py
ln -s ~/privateml/tensorflow/spdz/configs/gcp/server/role.py ~/role.py
ln -s ~/privateml/tensorflow/spdz/configs/gcp/server/pull.sh ~/pull.sh
ln -s ~/privateml/tensorflow/spdz/configs/gcp/server/monitor.sh ~/monitor.sh
ln -s ~/privateml/tensorflow/spdz/configs/gcp/server/tensorboard.sh ~/tensorboard.sh
