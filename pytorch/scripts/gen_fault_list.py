import sys
import importlib
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.gemmini.gemmini_config as conf


# Set this to true to generate fault lists for the systolic array. False to generate SW-only fault lists
USE_SA = True # True to generate both USE_SA and GL fault lists
#USE_SA = False # False to generate SW fault lists


# For the SA faults, select a Gemmini configuration

# OS configs
#CONFIG_KEY = "OSDIM4"
CONFIG_KEY = "OSDIM8"
#CONFIG_KEY = "OSDIM16"
#CONFIG_KEY = "OSDIM32"
#CONFIG_KEY = "OSDIM64"

# OS with TMR PEs
#CONFIG_KEY = "OSDIM8_TMR"

# OS configs with HDFIT FI instrumentation
#CONFIG_KEY = "OSDIM4_HDFIT"
#CONFIG_KEY = "OSDIM8_HDFIT"
#CONFIG_KEY = "OSDIM16_HDFIT"
#CONFIG_KEY = "OSDIM32_HDFIT"
#CONFIG_KEY = "OSDIM64_HDFIT"

# WS configs
#CONFIG_KEY = "WSDIM4"
#CONFIG_KEY = "WSDIM8"


if len(sys.argv) < 2:
    print(f"Syntax: python {sys.argv[0]} <model name>")
    exit(0)

MODEL = sys.argv[1]

print(f"Generating fault list for {MODEL}, {CONFIG_KEY}")

#MODEL="ResNet18"
#MODEL="ResNet50"
#MODEL="ResNeXt101_32X8D"
#MODEL="ResNeXt101_64X4D"
#MODEL="MobileNet_V2"
#MODEL="MobileNet_V3_Large"
#MODEL="GoogLeNet"
#MODEL="Inception_V3"
#MODEL="ShuffleNet_V2_X0_5"
#MODEL="ShuffleNet_V2_X1_0"
#MODEL="ShuffleNet_V2_X1_5"
#MODEL="ShuffleNet_V2_X2_0"
#MODEL="deit_tiny"
#MODEL="deit_small"

# import the layer shapes for the model
module_path = f"src.models.layers.{MODEL}"
TargetModel = importlib.import_module(module_path) # e.g., ResNet18, ResNet50,..., ShuffleNet_V2_X2_0

if USE_SA:
    conf.set_config(CONFIG_KEY)

vit_models = [
    "deit_tiny", "deit_small", "deit_base",
    "swin_tiny", "swin_small", "swin_base",
    "vit_base", "vit_large"]

VIT = MODEL in vit_models

if USE_SA:
    if VIT: 
        from src.flist import fault_list_vit as fl 
    else: 
        from src.flist import fault_list as fl 
else:
    if VIT: 
        from src.flist import fault_list_vit_sw as fl 
    else: 
        from src.flist import fault_list_sw as fl 

# set the fault list name. this is saved in fault_lists/<model>/FL_NAME
if USE_SA:
    FL_NAME=f"fl_os_dim_{conf.DIM}.csv"
else:
    FL_NAME = "fl_sw.csv"

# let this hard-coded name here for safety reasons (so that we do not overwrite any fault list by accident)
#FL_NAME="dummy_list_to_prevent_accidents.csv"

FAULTS_PER_LAYER_HW = 10000
FAULTS_PER_LAYER_SW = 10000

HW_TARGETS = [
    # data signals
    conf.SIGNAL['IN_A'], 
    conf.SIGNAL['IN_B'], 
    conf.SIGNAL['OUT_B'],
    conf.SIGNAL['OUT_C'],

    conf.SIGNAL['C1'], 
    conf.SIGNAL['C2'],

    # control signals
    conf.SIGNAL['SIG_PROPAG'], 
    conf.SIGNAL['SIG_VALID'], 
        ]

if VIT:
    DEPTH={
        "deit_tiny":  12,
        "deit_small": 12,
        "deit_base":  12
    }

    NUM_HEADS={
        "deit_tiny":  3,
        "deit_small": 6,
        "deit_base":  12
    }


def main():

    print(f"Generating fault list for model {MODEL}. Fault list: {FL_NAME}")
    
    if USE_SA:
        gen_fault_list_sa()
    else:
        gen_fault_list_sw()


# Generates Gemmini fault lists
def gen_fault_list_sa():
    full_list = []
    start_tag = 0

    print(f"Gemmini config: {conf.CONFIG_KEY}")
    
    for i in range(TargetModel.N_LAYERS):
        print(f"Generating fault list for layer {i}")
        if VIT:
            # for ViT, you can specify  depth:int=<>, num_heads:int=<>
            list_layer = fl.generate_fault_list(
                i, 
                TargetModel.shape_layer[i][0], 
                TargetModel.shape_layer[i][1], 
                start_tag, 
                possible_targets=HW_TARGETS,
                depth=DEPTH[MODEL], 
                num_heads=NUM_HEADS[MODEL],
                max_faults=FAULTS_PER_LAYER_HW)
        else:
            list_layer = fl.generate_fault_list(
                i, 
                TargetModel.shape_layer[i][0], 
                TargetModel.shape_layer[i][1], 
                start_tag, 
                possible_targets=HW_TARGETS,
                max_faults=FAULTS_PER_LAYER_HW)

        start_tag += len(list_layer)
        full_list.extend(list_layer)

    fn = f"{MODEL}/{FL_NAME}"

    fl.dump_fault_list(full_list, fn)


# Generates SW fault lists

def gen_fault_list_sw():
    full_list = []
    start_tag = 0

    for i in range(TargetModel.N_LAYERS):
        print(f"Generating fault list for layer {i}")
        
        if VIT:
            # SW fault lists for VIT (only activations so far, no weights)

            list_layer_0 = fl.generate_fault_list(
                i, 
                TargetModel.shape_layer_sw[i], 
                start_tag,  
                depth=DEPTH[MODEL], 
                num_heads=NUM_HEADS[MODEL],
                max_faults=FAULTS_PER_LAYER_SW)

            start_tag += len(list_layer_0)
            full_list.extend(list_layer_0)

        else:
            # SW fault list for CNNs

            # weights
            list_layer_0 = fl.generate_fault_list(
                i, 
                TargetModel.shape_layer_sw[i][0], 
                start_tag,
                max_faults=FAULTS_PER_LAYER_SW, 
                target=0)
            start_tag += len(list_layer_0)
            
            # activations
            list_layer_1 = fl.generate_fault_list(
                i, 
                TargetModel.shape_layer_sw[i][1], 
                start_tag, 
                max_faults=FAULTS_PER_LAYER_SW, 
                target=1)
            start_tag += len(list_layer_1)
            
            list_layer = list_layer_0 + list_layer_1

            full_list.extend(list_layer)

    fn = f"{MODEL}/{FL_NAME}"

    fl.dump_fault_list(full_list, fn)


if __name__ == '__main__':
    main()  
