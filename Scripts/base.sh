#!/bin/bash
#SBATCH -A p30137               # Allocation
#SBATCH -p short                # Queue
#SBATCH -t 1:00:00              # Walltime/duration of the job
#SBATCH -N 1                    # Number of Nodes
#SBATCH --mem=64G               # Memory per node in GB needed for a job. Also see --mem-per-cpu
#SBATCH --ntasks-per-node=24    # Number of Cores (Processors)
#SBATCH --mail-user=xander.hall@northwestern.edu  # Designate email address for job communications
#SBATCH --mail-type=END    # Events options are job BEGIN, END, NONE, FAIL, REQUEUE