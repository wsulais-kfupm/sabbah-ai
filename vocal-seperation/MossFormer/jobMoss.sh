#!/bin/sh


#SBATCH --job-name=Moss_Job
#SBATCH -o outMoss.out
#SBATCH --error=errMoss.err
#SBATCH --time=48:00:00
#SBATCH --mail-user=Mohammad.Shiekh@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --constraint=v100

        source /ibex/user/shiekhmf/MossFormer/Moss-env/bin/activate

python ./Libri2Mix_8k_train.py
