#!/bin/bash
conda activate py36
#$ -N job
#$ -V
#$ -cwd
#$ -l h_rt=02:00:00
#$ -l h_vmem=5G
#$ -pe threaded 1
#$ -j y
#$ -o ./logs/log_$JOB_NAME_$JOB_ID_$HOSTNAME.out

# Run: qsub -N <jobname> -V -cwd -l h_rt=02:00:00 -l h_vmem=5G -pe threaded 1 -j y -o <output_logfile> run_script.sh <script file>
# mkdir -p ./logs

STARTTIME=$(date +%s)
cmd="$@"
echo "Begin Date: `date +'%d/%m/%Y - %H:%M:%S'`"
echo "Node: [$HOSTNAME]"
echo "Working Directory: [$SGE_O_WORKDIR]"
echo "Job ID: [$JOB_ID]"
echo "Job name: [$JOB_NAME]"
echo "Command: $cmd"
echo

printf '\%.0s' {1..80}; echo
eval $cmd
JOB_STATUS=$?
printf '\%.0s' {1..80}; echo

echo
printf '=%.0s' {1..80}; echo
echo -e "Usage statistics for [$JOB_ID]:"
qstat -j $JOB_ID | grep -v -P '\\|sge_o_path'
printf '=%.0s' {1..80}; echo

echo "End Date: `date +'%d/%m/%Y - %H:%M:%S'`"
ENDTIME=$(date +%s)
ELAPSED=$(($ENDTIME - $STARTTIME))
echo "Execution duration: $((ELAPSED/3600))h:$(((ELAPSED/60)%60))m:$((ELAPSED%60))s"

if [ $JOB_STATUS -ne 0 ]; then
   echo "[i] Job has failed. Error!"
else
   echo "[i] Job finished successfully."
fi

#echo -e "\nUsage statistics for [$JOB_ID]:"
#qstat -j $JOB_ID

#qstat -f -j $JOB_ID
#set +o xtrace

