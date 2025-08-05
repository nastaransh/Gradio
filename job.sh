#!/bin/bash
#SBATCH --account=def-nast
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=12G
#SBATCH --time=00:10:00

# Load your environment

module load arrow cuda
source ENV/bin/activate

# Run your script
python chatbot.py
