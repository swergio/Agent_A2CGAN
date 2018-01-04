#!/bin/bash
tensorboard --logdir=/usr/src/app/sources/Log &
python -u /usr/src/app/run.py