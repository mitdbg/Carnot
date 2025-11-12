#!/bin/bash

# Job Flags
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:l40s:1

# Set up environment
source venv/bin/activate

# Run your application
python3 index_chroma_limited.py