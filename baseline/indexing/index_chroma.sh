#!/bin/bash

# Job Flags
#SBATCH -p mit_normal
#SBATCH -c 4

# Set up environment
source venv/bin/activate

# Run your application
python3 index_chroma_verbose.py