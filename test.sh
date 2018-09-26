#!/bin/bash

python2.7 test.py --graph=frozen.pb --labels=speech_commands_train/conv_labels.txt --wav=../speech_dataset/left/a5d485dc_nohash_0.wav
echo "--------------------------------------------------------------------------------------"
python2.7 test.py --graph=frozen.pb --labels=speech_commands_train/conv_labels.txt --wav=../speech_dataset/right/0a2b400e_nohash_0.wav
echo "--------------------------------------------------------------------------------------"
python2.7 test.py --graph=frozen.pb --labels=speech_commands_train/conv_labels.txt --wav=../speech_dataset/up/0ab3b47d_nohash_0.wav
echo "--------------------------------------------------------------------------------------"
python2.7 test.py --graph=frozen.pb --labels=speech_commands_train/conv_labels.txt --wav=../speech_dataset/down/525eaa62_nohash_2.wav
echo "--------------------------------------------------------------------------------------"
python2.7 test.py --graph=frozen.pb --labels=speech_commands_train/conv_labels.txt --wav=../speech_dataset/yes/0a9f9af7_nohash_2.wav
echo "--------------------------------------------------------------------------------------"
python2.7 test.py --graph=frozen.pb --labels=speech_commands_train/conv_labels.txt --wav=../speech_dataset/no/5ebc1cda_nohash_1.wav
echo "--------------------------------------------------------------------------------------"
python2.7 test.py --graph=frozen.pb --labels=speech_commands_train/conv_labels.txt --wav=../speech_dataset/on/0ba018fc_nohash_3.wav
echo "--------------------------------------------------------------------------------------"
python2.7 test.py --graph=frozen.pb --labels=speech_commands_train/conv_labels.txt --wav=../speech_dataset/off/0a196374_nohash_2.wav
echo "--------------------------------------------------------------------------------------"
python2.7 test.py --graph=frozen.pb --labels=speech_commands_train/conv_labels.txt --wav=../speech_dataset/stop/9be15e93_nohash_4.wav
echo "--------------------------------------------------------------------------------------"
python2.7 test.py --graph=frozen.pb --labels=speech_commands_train/conv_labels.txt --wav=../speech_dataset/go/0a2b400e_nohash_3.wav
echo "--------------------------------------------------------------------------------------"
python2.7 test.py --graph=frozen.pb --labels=speech_commands_train/conv_labels.txt --wav=../speech_dataset/cat/29dce108_nohash_0.wav
echo "--------------------------------------------------------------------------------------"

