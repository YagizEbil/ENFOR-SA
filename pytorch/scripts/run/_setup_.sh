source "experiment.sh"

# fixed parameters that should not be changed
MODE_SEQUENTIAL=0
MODE_PARALLEL=1
MODE_DAC=2

root_report_path="reports"

init_params(){
    simul_mode=$1
    if [ $simul_mode -eq $MODE_SEQUENTIAL ]; then
        declare -g exp_descr=$experiment_description
        declare -g mode_folder="sequential"
    elif [ $simul_mode -eq $MODE_PARALLEL ]; then
        declare -g exp_descr=$experiment_description
        declare -g mode_folder="parallel"
    elif [ $simul_mode -eq $MODE_DAC ]; then
        declare -g exp_descr=$experiment_description
        declare -g mode_folder="dac"
    else
        echo "[Error]: Invalid simulation mode"
        exit
    fi

    # buils the report directories in case they still don't exist there
    # this is a fixed pattern
    declare -g base_dir="$root_report_path/$model/$experiment/$mode_folder"

    build_dir "$base_dir/batch/"
    build_dir "$base_dir/trace/"

    if [ $simul_mode -eq $MODE_PARALLEL ]; then
        build_dir "$base_dir/nodes/"
    fi

    if [ $simul_mode -eq $MODE_DAC ]; then
        build_dir "$base_dir/nodes/"
    fi
}

# creates the reports destination folders if not available
build_dir(){
if [ ! -d "$1" ]; then
    mkdir -p "$1"
fi 
}

# writes the experiment description in a log file
write_notes_start(){
# experiment description written in the $experiment folder

time_now=$(date)

# creates the "experiment.log" file
experiment_info="
<$time_now>
    Model:       $model
    Machine:     $(hostname)
    Exp. file:   $experiment_file
    Gemmini:     $gemmini_config_key
    Fault model: $fault_model
    Notes:       $exp_descr
    Alias:       $sim_alias
    Threads:     (inter,intra)=($inter_threads, $intra_threads)
    Fault list:  $fault_list
    Injections:  $injections
    Seed:        $seed
    Batch size:  $batch_size
    Batches:     $nbatches
    Layers:      ${target_layers[@]}"
echo "$experiment_info" >> "$base_dir/experiment.log"
}

#Layers:     [$layer_start, $layer_end]"

# writes the experiment description in a log file
write_notes_end(){
time_now=$(date)
elapsed=$1
echo "    Elapsed time: $elapsed seconds" >> "$base_dir/experiment.log"
echo "    Return code:  $2"               >> "$base_dir/experiment.log" 
echo "</$time_now>" >> "$base_dir/experiment.log"
}


