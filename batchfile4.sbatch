#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=test
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --mem=15GB
###SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load python/intel/2.7.12 pytorch/0.2.0_1 protobuf/intel/3.1.0 spyder/3.1.4
module load torchvision/0.1.8

python main.py --data data
# python evaluate.py --data data --model model_75.pth --outfile out1.csv
python evaluate.py --data data --model model_76.pth --outfile out2.csv
# python evaluate.py --data data --model model_77.pth --outfile out3.csv
# python evaluate.py --data data --model model_78.pth --outfile out4.csv
# python evaluate.py --data data --model model_79.pth --outfile out5.csv
# python evaluate.py --data data --model model_80.pth --outfile out1.csv
# python evaluate.py --data data --model model_81.pth --outfile out2.csv
# python evaluate.py --data data --model model_82.pth --outfile out3.csv
# python evaluate.py --data data --model model_83.pth --outfile out4.csv
# python evaluate.py --data data --model model_84.pth --outfile out5.csv
# python evaluate.py --data data --model model_85.pth --outfile out6.csv


