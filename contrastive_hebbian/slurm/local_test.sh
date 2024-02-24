N_RUNS=20
for ((i=0; i < $N_RUNS; i++))
do
    python flexnet_slurm.py --run-id $i
done
