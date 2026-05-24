models=(
        #"ResNet18"
        #"ResNet50"
        "ResNeXt101_32X8D"
        #"ResNeXt101_64X4D"
        #"MobileNet_V2"
        #"MobileNet_V3_Large"
        #"GoogLeNet"
        #"Inception_V3"
        #"ShuffleNet_V2_X0_5"
        #"ShuffleNet_V2_X2_0"
        #"deit_tiny"
        #"deit_small"
      )

for ((i=0; i<${#models[@]}; i++)); do
  python scripts/gen_fault_list.py "${models[i]}" 
done
