#SBATCH --job-name="Calculate Possible FU Oris"       # Name of job

echo deploying job ...

# add a project directory to your PATH (if needed)
export PATH=$PATH:/projects/p30137/xhall/

# load modules you need to use
module purge all
module load python/anaconda3.6
source activate pymc3_env