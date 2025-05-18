#!/bin/bash
#SBATCH --chdir={{ directory }}
#SBATCH --partition=single
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --time=120:00:00
#SBATCH --mem=50gb
#SBATCH --ntasks-per-node=16
#SBATCH --export=NONE
#SBATCH --mail-user={{ email }}
#SBATCH --mail-type=BEGIN,END

{{ module_loads }}

cd {{ directory }}

cmd="{{ command }}"

if compgen -G "*{{ gro_out_file }}*gro*" > /dev/null ; then
    echo "JOB FINISHED"
    exit
fi

# count the number of slurm out files
num=$( ls -1 *out* | wc -l )
if [ $num -gt {{ max_out_files }} ] ; then
    echo "Too many out files exiting."
    exit
fi


if [!  -e {{ cpt_out_file }}.cpt ]; then
echo " ### Initial submit. Starting job. ### "
$cmd
else
echo " ### Continuation ### "
cp {{ cpt_out_file }}.cpt {{ cpt_out_file }}-$( date --iso-8601=seconds ).cpt.back
$cmd -cpi {{ cpt_out_file }}.cpt
fi

if ! compgen -G "*{{ gro_out_file }}*gro*" > /dev/null ; then
echo "Job not finished."
echo "Submiting..."
sbatch job.sh
fi
