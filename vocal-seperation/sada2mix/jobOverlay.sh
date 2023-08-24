#!/bin/sh


#SBATCH --job-name=Overlay_Job
#SBATCH -o outLay.out
#SBATCH --error=errLay.err
#SBATCH --time=48:00:00
#SBATCH --mail-user=Mohammad.Shiekh@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --mem=64G
#SBATCH --cpus-per-task=6

        source /ibex/user/shiekhmf/MossFormer/Moss-env/bin/activate

python ./overlaying.py
