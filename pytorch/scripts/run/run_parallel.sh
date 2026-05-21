#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_setup_.sh"

init_params $MODE_PARALLEL
write_notes_start
start_time_s=$(date +%s)
return_code_total=0

for layer in "${target_layers[@]}"; do
    python3 main.py \
        --model $model \
        --fmodel $fault_model \
        --config $gemmini_config_key \
        --batches $nbatches \
        --bsize $batch_size \
        --layer $layer \
        --faultlist $fault_list \
        --injections $injections \
        --output $base_dir \
        --alias $sim_alias \
        --th_inter $inter_threads \
        --th_intra $intra_threads \
        --seed $seed \
        --tree \
        --prune \

    #the script's return code in sys.exit(<return code>)
    ret=$?
    return_code_total=$((return_code_total + ret))
done

end_time_s=$(date +%s)
elapsed=$((end_time_s - start_time_s))
write_notes_end $elapsed $return_code_total
echo "Script (run_parallel.sh) finished after $elapsed seconds"

#gdb --args python main.py -model ResNet18 -config OSDIM8 -output reports/ResNet18/debug/parallel -bsize 1 -batches 1 -layer 0 -injections 100 -faultlist fl_os_dim_8.csv -tree True -prune False -th_inter 2 -th_intra 2 -alias B20-b32-i500 -gold False -seed 0 
#run
#backtrace


