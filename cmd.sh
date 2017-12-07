#!/bin/bash
tensorboard --logdir=/usr/src/app/log &
python -u /usr/src/app/run.py