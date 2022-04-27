#!/bin/bash

#$ -M yyan6@nd.edu   # Email address for job notification
#$ -m abe            # Send mail when job begins, ends and aborts
#$ -q long           # Specify queue
#$ -N HOG_SGD       # Specify job name

conda activate nn1      # Required modules

python sgd.py
