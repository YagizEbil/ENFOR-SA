# Simulation parameters 
# This is an experiment script that defines experiment parameters. Among other things, this script defines:
# 	$experiment: An experiment name (will be the report folder name);
# 	$gemmini_config_key: A systolic array configuration (examples provided);
# 	$fault_model: The fault injection mode (options are “sw”, “rtl”, and “gl”);
# 	$experiment_description: a brief description of the experiment;
# 	$sim_alias: used as the part of the report files names;
# 	$fault_list: A fault list file (stored in the folder python/fault_lists);
# 	$injections: The number of faults to inject;
# 	$batch_size: The input batch size (ideal depends on the PC);
# 	$nbatches: The number of batches to inject.

#
# The experiment report files. Those are placed in the folder reports/$model/$experiment. Each experiment type should be placed in its own folder
#
experiment="exp-activations-SW-v1"

#
# The Gemmini configuration key (only valid for RTL/GL injection). See all available config keys in gemmini_config.py
#
gemmini_config_key="N/A" 

#
# The desired fault model
#
fault_model="sw"

#
# An experiment description for bookkeeping
#
experiment_description="Standard sequential injection approach at SW. Injection in the activations"

#
# A simulation alias for bookkeeping - this is attatched to the report file names
#
sim_alias="simulation-xxx"

#
# The fault list. It must be located in the path fault_lists/$model
#

# SW injection
fault_list="fl_sw.csv"

#
# The number of faults to be injected for each input in each target layer
#
injections=100

#
# The number of inputs per batch (16 is the best case for my pc)
#
batch_size=32

#
# The number batches
#
nbatches=20

#
# A simulation seed (this goes to the report file name)
#
seed=1

#
# The number of PyTorch inter and intra op threads (see https://docs.pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html)
# (set a huge value, e.g., 1000, to adjust to the max possible number)
#
inter_threads=16
intra_threads=16

#
# The model name and layers come from run.sh
#
model=$1
shift # remove $1, so now $1..$N are the layers
target_layers=("$@") # capture all remaining args as an array

#
# This script's name
#
experiment_file="${BASH_SOURCE[0]}"
