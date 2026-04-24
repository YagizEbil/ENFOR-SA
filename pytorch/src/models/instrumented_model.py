import torch
import re
import types
import time
import importlib

from torch._ops import ops
from src.models.base_model import BaseModel
from src.utils import utils as u
from src import definitions as defs

if defs.FI_GEMM:
    from src.gemmini import gemmini_config as conf
    from src.gemmini import gemmini_interface
    from src.conv import custom_conv_gemmini as custom_conv
    from src.models import masking_track 
else:
    from src.conv import custom_conv_sw as custom_conv


# the instrumented model with replaced layers (e.g., conv2d is run as custom_conv.conv(...))
class InstrumentedModel(BaseModel):
    def __init__(self, model_name):
   
        super().__init__(model_name)

        print(f"{u.Co['fg'][65]}[InstrumentedModel]:{u.R}")
        print(f"- DNN model: {model_name}")
        print(f"- FI layer:  {defs.TARGET_LAYER}")
        #print(f"- FI layer:  {TargetModel.conv_layers_names[defs.TARGET_LAYER]}/{defs.TARGET_LAYER}")
        #print(f"- FI mode:   {'Parallel' if defs.TREE_FI_MODE else 'Sequential'}")
        print(f"- FI level:  {'RTL' if defs.FI_GEMM else 'Software'}")

        # if using RTL, loads the gemmini device
        if defs.FI_GEMM:
            gemmini_interface.gemmini_device = gemmini_interface.GemminiOS(conf.CONFIG_KEY)

        if defs.VIT: # ViT intrumentation for FI
            if defs.FI_GEMM:
                self.count_replaced_conv_layers = instrument_vit_block(self.model)
            self.gemm_conv = None # we only use this for CNNs

        else: # conv. layer replacement for FI 
            if defs.FI_GEMM:
                self.gemm_conv = custom_conv.CustomConv()  # this object is only usefull for CNNs

            self.count_replaced_conv_layers = instrument_cnn_layers(self)  # the number of replaced layers

        self.relu = torch.nn.ReLU()
        self.is_golden_mode = False

        if defs.FI_GEMM:
            self.stats_gemm_msk = masking_track.GemminiMaskingTrack(None)
        print()


    """ the conv forward pass """
    def forward_quantized_conv(self, layer, input_tensor):
        if not defs.FI_GEMM: # SW Injection only
            return custom_conv.conv_batch_sw_fi(layer, input_tensor)

        if not self.gemm_conv.prepared: 
            self.gemm_conv.load_params(layer)
            self.gemm_conv.prepared = True

        ret_tensor, mask_levels = self.gemm_conv.conv_batch(layer, input_tensor)
        self.stats_gemm_msk = masking_track.GemminiMaskingTrack(mask_levels)

        return ret_tensor


    def hook_layer(self, layer, to_new_forward_type):
        self_ = self  # the InstrumentedModel instance

        def new_forward_conv(self, *inputs, **kwargs): # here, 'self' refers to a layer model, not a InstrumentedModel instance
            return self_.forward_quantized_conv(layer, *inputs)

        def new_forward_conv_relu(self, *inputs, **kwargs):
            r = new_forward_conv(layer, *inputs, **kwargs)
            return self_.relu(r)

        if to_new_forward_type == custom_conv.CONV:
            layer.forward = types.MethodType(new_forward_conv, layer)
        
        elif to_new_forward_type == custom_conv.CONV_RELU:
            layer.forward = types.MethodType(new_forward_conv_relu, layer)
        
        else:
            raise "Failed: Unknown new forward type"


if defs.VIT:
    from src.ivit.models.quantization_utils.quant_modules_fi_adapter import QuantMatMulGemmini
    from src.ivit.models.quantization_utils.quant_modules_fi_adapter import QuantLinearGemmini

    def instrument_vit_block(model):
        #return # [Debug only]

        if not defs.FI_GEMM:
            raise("Blocks should only be replaced for RTL injection")

        if defs.TARGET_LAYER == 6: # layer 6: Classifier head - type QuantLinear()
            """
                Some notes on this layer: when injecting here, the output logits show very little variance when compared to the other layers
            """
            print("Replaced transformer layer 6")
            print(model.head)

            model.head = QuantLinearGemmini(
                in_features=model.head.in_features, 
                out_features=model.head.out_features, 
                bias=True) 
            return 

        # replaces all blocks. however, only the target one is computed in Gemmini
        # loops through each block
        depth = len(model.blocks)  # nblocks = depth this comes from the VisionTransformer model depth (12 for most models. 24 for vit large)
        
        for blk_id in range(depth):
            block = model.blocks[blk_id]
            attention = block.attn

            """ inside each block (deit models), there are the following 6 types of layers """

            if defs.TARGET_LAYER == 0: # layer 0: i think this computes Q=Wq*V, K=Wk*V, V=Wv*V - type QuantLinear()
                print("Replaced transformer layer 0")
                print(block.attn.qkv)
                
                attention.qkv = QuantLinearGemmini(
                    in_features=attention.qkv.in_features, 
                    out_features=attention.qkv.out_features, 
                    bias=True)
                
            elif defs.TARGET_LAYER == 1: # layer 1: computes x=QxK  - type QuantMatMul() 
                # a lot of zeros - check if there was an activation in the prev. layer
                print("Replaced transformer layer 1")
                print(attention.matmul_1)
                attention.matmul_1 = QuantMatMulGemmini()

            elif defs.TARGET_LAYER == 2: #  layer 2: computes activation(x)*V - type QuantMatMul()
                print("Replaced transformer layer 2")
                print(attention.matmul_2)
                attention.matmul_2 = QuantMatMulGemmini()
            
            elif defs.TARGET_LAYER == 3: # layer 3: projection - possibly eq. 12.3 the attention of each token to each token? - type QuantLinear()
                print("Replaced transformer layer 3")
                print(attention.proj)

                attention.proj = QuantLinearGemmini(
                    in_features= attention.proj.in_features, 
                    out_features=attention.proj.out_features, 
                    weight_bit=8, 
                    quant_mode='symmetric')
            
            elif defs.TARGET_LAYER == 4: #layer 4 (fc1): the block has an extra 'mlp' before the "Classifier head" - i do not know why - type QuantLinear()
                print("Replaced transformer layer 4")
                print(block.mlp.fc1)
                
                block.mlp.fc1 = QuantLinearGemmini(
                    in_features=block.mlp.fc1.in_features, 
                    out_features=block.mlp.fc1.out_features)
            
            elif defs.TARGET_LAYER == 5: # layer 5 (fc2): the block has an extra 'mlp' before the "Classifier head" - i do not know why - type QuantLinear()
                print("Replaced transformer layer 5")
                print(block.mlp.fc2)
                
                block.mlp.fc2 = QuantLinearGemmini(
                    in_features=block.mlp.fc2.in_features, 
                    out_features=block.mlp.fc2.out_features)

            else:
                raise("Invalid layer replacement")


else: # CNNs only
    
    def instrument_cnn_layers(model):
        # import the model in src/layers. This is only used to get the layer names
        module_path = f"src.models.layers.{defs.MODEL_NAME}"
        TargetModel = importlib.import_module(module_path) # e.g., ResNet18, ResNet50,..., ShuffleNet_V2_X2_0

        print(f"- Layer: {TargetModel.conv_layers_names[defs.TARGET_LAYER]}")

        #[COLLECT LAYER SHAPES]
        REPL_ALL = False # set this to true to replace all conv layers at once (used to get the input shapes of each layer to build the fault lists)

        count_replaced_conv_layers = 0

        #layers_to_repl =  ['layer1.0.conv2'] # use the full layer name (the object name, not the class name!) for the layer to be replaced
        layers_to_repl = [TargetModel.conv_layers_names[defs.TARGET_LAYER]] # replaces only one layer passed as parameter (--layer)

        #print("conv_layers_names={") #  [COLLECT LAYER SHAPES]

        for name, layer in model.model.named_modules():  # named_children() would require recursion
            if isinstance(layer, torch.nn.quantized.Conv2d):
                #print(f"{count_replaced_conv_layers}:'{name}',") # uncomment to generate the dict. of layer names (use REPL_ALL=True)
                if name in layers_to_repl or REPL_ALL:
                    str_module_name = str(layer)
                    if re.match(r"QuantizedConvReLU2d.*", str_module_name) is not None:
                        print(f"- Replaced: {str_module_name}")
                        #print(f"{count_replaced_conv_layers}:'{name}',") #[COLLECT LAYER SHAPES]
                        model.hook_layer(layer, custom_conv.CONV_RELU)
                        
                        if defs.FI_GEMM:
                            model.gemm_conv.conv_type = custom_conv.CONV_RELU

                        count_replaced_conv_layers += 1
                    
                    elif re.match(r"QuantizedConv2d.*", str_module_name) is not None:
                        print(f"- Replaced: {str_module_name}")
                        #print(f"{count_replaced_conv_layers}:'{name}',") #[COLLECT LAYER SHAPES]
                        model.hook_layer(layer, custom_conv.CONV)
                        
                        if defs.FI_GEMM:
                            model.gemm_conv.conv_type = custom_conv.CONV
                        
                        count_replaced_conv_layers += 1

                    else:
                        raise ValueError(f"Unknown conv module! {str_module_name}")

        return count_replaced_conv_layers


