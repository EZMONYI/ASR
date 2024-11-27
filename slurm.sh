#!/bin/bash

#SBATCH --job-name=qwen2annotate
#SBATCH --output=./logs/%j.txt
#SBATCH --ntasks=1
#SBATCH --mem=8g
#SBATCH --partition=audio_mt_spk
#SBATCH --gres=gpu:8

source ~/.bashrc
conda activate asr

python qwen2audio_annotator.py -i data/test_data_1126 -o output 
