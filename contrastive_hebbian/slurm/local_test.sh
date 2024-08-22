# This is before Cosyne
N_RUNS=9
for ((i=0; i < $N_RUNS; i++))
do
    python flexnet_slurm.py --run-id $i
done

# Refactored code, no curriculum, small angle

