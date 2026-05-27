#!/usr/bin/env bash

# Enable the models to inject here
models=(
        "ResNet18"
        #"ResNet50"
        #"ResNeXt101_32X8D"
        #"ResNeXt101_64X4D"
        #"MobileNet_V2"
        #"MobileNet_V3_Large"
        #"GoogLeNet"
        #"Inception_V3"
        #"ShuffleNet_V2_X0_5"
        #"ShuffleNet_V2_X2_0"
        #"deit_tiny"
        #"deit_small"
        #"deit_base"
      )

ResNet18_target_layers=(0 1 2 10 11 12 17 18 19)
ResNet50_target_layers=(0 1 2 5 10 15 20 25 30 35 40 45 50 51 52)
ResNeXt101_32X8D_target_layers=(0 1 2 20 30 40 50 60 70 80 90 101 102 103)
ResNeXt101_64X4D_target_layers=(0 1 2 20 30 40 50 60 70 80 90 101 102 103)
ShuffleNet_V2_X0_5_target_layers=(0 1 2 10 15 20 25 30 35 40 45 50 54 55) 
ShuffleNet_V2_X2_0_target_layers=(0 1 2 10 15 20 25 30 35 40 45 50 54 55) 
MobileNet_V2_target_layers=(0 1 2 10 20 30 40 49 50 51)
MobileNet_V3_Large_target_layers=(0 1 2 10 20 30 40 50 59 60 61) 
GoogLeNet_target_layers=(0 1 2 10 15 20 25 30 35 40 45 50 54 55 56) 
Inception_V3_target_layers=(0 1 2 10 20 30 40 50 60 70 80 90 92 93) 

# deit layers (all variants)
deit_layers=(0 1 2 3 4 5 6)

# [Debug only]
debug_layers=(0)

# Enable the required injection layers here, for each model enabled in the 'models' array above 
# this array must be consistent with the used models above

target_layers=(
                #ResNet18_target_layers
                #ResNet50_target_layers
                #ResNeXt101_32X8D_target_layers
                #ResNeXt101_64X4D_target_layers
                #MobileNet_V2_target_layers
                #MobileNet_V3_Large_target_layers
                #GoogLeNet_target_layers
                #Inception_V3_target_layers
                #ShuffleNet_V2_X0_5_target_layers
                #ShuffleNet_V2_X2_0_target_layers
                #deit_layers
                #deit_layers
                #deit_layers
                debug_layers
            )
#
# Points to the layers declared above, for each model
#
declare -n used_layers="${target_layers}"

#
# Runs the sequential fault injection mode
#
for ((i=0; i<${#models[@]}; i++)); do
  #
  # Runs the sequential fault injection mode
  #
  ./scripts/run/run_sequential.sh "${models[i]}" "${used_layers[@]}" 
done

#
# Generates the app profile. Run this line, and then "cd ./profiler" and "./gen-svg.sh" (the svg output will be saved in profiler/svg)
#
#for ((i=0; i<${#models[@]}; i++)); do
#  ./scripts/run/prof.sh "${models[i]}" "${used_layers[@]}"
#done
