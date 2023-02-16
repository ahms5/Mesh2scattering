#!/usr/local_rwth/bin/zsh
#SBATCH --account=rwth1245

### Job name
#SBATCH --job-name=hpc_sample_3

### File / path where STDOUT will be written, the %J is the job id
#SBATCH --output=hpc_sample_3-%J.log

### Request the time you need for execution. The full format is D-HH:MM:SS
### You must at least specify minutes or days and hours and may add or
### leave out any other parameters % 10 hours
#SBATCH --time=00-03:00:00

### Request memory you need for your job in MB
#SBATCH --mem-per-cpu=4000M

### Request number of hosts
#SBATCH --nodes=1

### Request number of CPUs
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=1

### E-Mail notification
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ahe@akustik.rwth-aachen.de

#SBATCH --array=1


### Change to the work directory
cd $HOME/mesh2scattering/hpc

### load modules and execute
module load DEVELOP
module load gcc/11

export DISPLAY="localhost:0.0"


### start non-interactive batch job

cd ../sample/NumCalc/source_${SLURM_ARRAY_TASK_ID}
../../../../NumCalc -istart 3 -iend 3 >NC3-3_log.txt
