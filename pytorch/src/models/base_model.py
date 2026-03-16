import torch
import time
import os
import warnings

from collections import defaultdict
from typing import List, Optional

from src.models import common
from src.conv import cached_tensors as tcache
from src.utils import dataset_loader as dataloader
from src import definitions as defs
from src.flist.fl import fl

#from . import forward_ResNet18 as fw_ResNet18

warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", message="must run observer before calling calculate_qparams")

# encapsulates a base dnn model ands provides basic model inference and tensor manipulation facilities
class BaseModel(torch.nn.Module):

    def __init__(self, model_name):
        super(BaseModel, self).__init__()

        if defs.VIT: # ViTs
            self.model = common.QUANTIZED_MODELS_VIT(model_name)(pretrained=True, num_classes=1000, drop_rate=0, drop_path_rate=0.1)

        else: # CNNs
            weights_work = common.QUANTIZED_MODELS[model_name][0]
            model_work   = common.QUANTIZED_MODELS[model_name][1]
            self.model   = model_work(weights=weights_work, quantize=True)

        self.model.eval()

        BaseModel.input_batch_indices: List[int] = None
        self.is_golden = True
        self.calibrate = False
    

    def run_inference(self, inputs, fault=None):
        fl.next_fault = fault

        with torch.no_grad():  # No need to compute gradients
            # Get the model's output
            output_logits = self(inputs)

            if defs.CUDA:
                torch.cuda.synchronize()

            return output_logits


    # runs a new batch inference (both golden or faulty models can call this). this stores model-dependent results
    def run_batch_inference(self):
        if len(BaseModel.input_batch): # when pruning the inputs in the tree mode, the batch maybe completely erased
            with torch.no_grad():  # No need to compute gradients
                # Get the model's output
                self.output_logits = self(BaseModel.input_batch)

                if defs.CUDA:
                    torch.cuda.synchronize()

                # Apply softmax to get probabilities
                self.probabilities = torch.nn.functional.softmax(self.output_logits, dim=1)
     
                # top5_classes.values contains the score probabilities - top5_classes.indices containts the associated classification indices
                self.top5_classes = torch.topk(self.probabilities, 5, dim=1)

                # the top 5 labels
                self.top5_classes_indices = self.top5_classes.indices      

                # the top-5 scores  
                self.top5_classes_values = self.top5_classes.values  # shape ([16,5])
                #top5_classes_values, top5_classes_indices = torch.topk(self.probabilities, k=5, dim=1)

                # Get the predicted top-1 label for each input in the batch
                self.predicted_labels = torch.argmax(self.probabilities, dim=1)
    
                return self.predicted_labels, self.top5_classes, self.output_logits

        return None, None, None


    # loads an input batch and runs the golden mode. stores batch-related info (static class members) and golden mode information
    def run_golden_batch(self, batch_indices: List[int]):
        if not self.is_golden:
            print("[Error]: run_golden_batch() called from non-golden model")
            exit(0)

        # early‑out if unchanged
        if not self.calibrate and batch_indices == BaseModel.input_batch_indices:
            return True

        if not batch_indices: # empty list - do nothing else (maybe the tree approach pruned all inputs)
            return False

        #custom_conv.input_ids = batch_indices[:] # moved to experiment_sequential.py

        # returns a batch of input tensors with their respective expected top1 labels
        BaseModel.input_batch, BaseModel.ground_truth_labels = dataloader.get_input_batch(batch_indices)
        BaseModel.input_batch = BaseModel.input_batch.to(defs.DEVICE)
        BaseModel.ground_truth_labels = BaseModel.ground_truth_labels.to(defs.DEVICE)

        # reads a toy input batch from the examples in folder toy_inputs
        #BaseModel.input_batch, BaseModel.ground_truth_labels = dataloader.load_toy_input(batch_indices[0])
        
        # the batch of inputs (list of indexes to the imagenet dataset inputs)
        BaseModel.input_batch_indices = batch_indices[:] # shallow copy is enough
        BaseModel.input_batch_size = len(batch_indices)
        BaseModel.input_batch_size_full = len(batch_indices)

        # runs the batch in golden mode
        _, _, _ = self.run_batch_inference()

        # counts the number of inputs in the batch that are mispredicted (wrong top1 class)
        count_gold_top_1_mispredicted = torch.sum(self.predicted_labels != BaseModel.ground_truth_labels).item()

        # computes the batch top1 accuracy
        self.batch_top1_accuracy = (BaseModel.input_batch_size - count_gold_top_1_mispredicted)/BaseModel.input_batch_size
        #print(f"Batch top1 accuracy: {100*self.batch_top1_accuracy:.4f}% (size: {BaseModel.input_batch_size})")

        # empties the tensor LUTs - must call this when running multiple inputs in the same campaign to force the LUT tensors to be loaded again to the new input
        tcache.clear_luts()


    def layer_names(self): # Returns a list of layer names.
        return [name for name, _ in self.model.named_modules()]


    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)


