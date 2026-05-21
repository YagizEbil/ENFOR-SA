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

def count_quantized_weights(model):
    total = 0
    for name, module in model.named_modules():
        if hasattr(module, "weight"):
            w = module.weight()
            if w is not None:
                total += w.numel()
        if hasattr(module, "bias") and module.bias() is not None:
            total += module.bias().numel()
    return total


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

        self.is_golden = True

        BaseModel.input_batch_indices: List[int] = None

        self.top1_conf = defaultdict(int)
        self.top2_conf = defaultdict(int)
        self.conf_gap  = defaultdict(int)

        # Use all layers of the base model except the final fully connected (fc) layer
        # this would be used for fastforwarding the model
        #self.features = torch.nn.Sequential(*list(self.model.children())[:-1])
        #self.features = torch.nn.Sequential(*list(self.model.children())[:]) # take all layers
        #self.n_layers = len(self.features)

        #print(self.features)
        #exit(0)

        self.time_samples_forward_pass = [] # [Debug only]: stores the execution times of the forward pass
        self.calibrate = False
    

    def run_inference(self, inputs, fault=None):
        #if fault != None:
        fl.next_fault = fault

        with torch.no_grad():  # No need to compute gradients
            # Get the model's output
            #self.output_logits = self.model(BaseModel.input_batch) # this wont call the forward pass!!
            output_logits = self(inputs)

            if defs.CUDA:
                torch.cuda.synchronize()

            return output_logits


    # every type of variable that is shared between faulty and golden models is static (i.e, BaseModel.var):
    #   ex: input batches, input batch ids, ground truth labels (these are set whenever a new golden run is performed)
    # other types of fields are not static (these are set whenever a new inference is performed)
    #   ex: classification labels, scores, accuracy, etc.

    # runs a new batch inference (both golden or faulty models can call this). this stores model-dependent results
    def run_batch_inference(self):
        if len(BaseModel.input_batch): # when pruning the inputs in the tree mode, the batch maybe completely erased
            with torch.no_grad():  # No need to compute gradients
                # Get the model's output
                #self.output_logits = self.model(BaseModel.input_batch) # this wont call the forward pass!!
                self.output_logits = self(BaseModel.input_batch.float())

                if defs.CUDA:
                    torch.cuda.synchronize()

                # TODO: we do not need to compute softmax to get the classification
                # softmax should be used only to compute the model confidence

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
                #self.predicted_labels = torch.argmax(self.output_logits, dim=1)

                """
                # TODO: this should give me equivalent results as the code above, and faster
                # the results are not matching
                
                self.top5_classes = torch.topk(self.output_logits, 5, dim=1)
                self.top5_classes_indices = self.top5_classes.indices[:, 0]
                self.top5_classes_values = self.top5_classes.values[:, 0]
                self.predicted_labels = self.top5_classes_indices
                """

                #"""
                if True: #defs.TREE_FI_MODE: # this is also used to compute the input crit. gaps
                    if self.is_golden:
                        # computes the conf. gap between top1 and top2 confidence levels
                        for i in range(len(BaseModel.input_batch)):
                            self.top1_conf[BaseModel.input_batch_indices[i]] = self.top5_classes_values[i][0].item()
                            self.top2_conf[BaseModel.input_batch_indices[i]] = self.top5_classes_values[i][1].item()
                            self.conf_gap[BaseModel.input_batch_indices[i]] = (self.top1_conf[BaseModel.input_batch_indices[i]] - self.top2_conf[BaseModel.input_batch_indices[i]])
                #"""
                return self.predicted_labels, self.top5_classes, self.output_logits

        return None, None, None


    # loads an input batch and runs the golden mode. stores batch-related info (static class members) and golden mode information
    def run_golden_batch(self, batch_indices: List[int]):
        if not self.is_golden:
            print("[Error]: run_golden_batch() called from non-golden model")
            exit(0)

        # early‑out if unchanged
        if not self.calibrate and batch_indices == BaseModel.input_batch_indices and not self.is_golden:
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


        # stores the original "full" batch data
        # when running the tree mode with input pruning, edit_tensors will take slices of the full batch for different fault nodes
        if defs.PRUNE_INPUTS:
            BaseModel.input_batch_full = BaseModel.input_batch.clone().detach()
            BaseModel.input_batch_indices_full = BaseModel.input_batch_indices.copy()  # the dataset input indices
            BaseModel.ground_truth_labels_full = BaseModel.ground_truth_labels.clone().detach()
            BaseModel.top1_labels_full = self.predicted_labels.clone().detach()
            BaseModel.top5_classes_indices_full = self.top5_classes.indices.clone().detach() #  the classificaton indexes (labels)
            BaseModel.top5_classes_values_full  = self.top5_classes.values.clone().detach()
            BaseModel.top_5_logits_full  = self.output_logits.clone().detach()

        # empties the tensor LUTs - must call this when running multiple inputs in the same campaign to force the LUT tensors to be loaded again to the new input
        tcache.clear_luts()


    # edit_tensors: receives a list of new batch indexes, and returns only the tensors and labels for the associated index 
    # example: the original full input indices are [a, b, c, d, e] -> the indexes after pruning are [b, d]
    #          this function thus returns the tensors and labels of indices [1,3] from the full batch set
    # this is to avoid having to re-run the golden inference againg for the pruned batch indexes
    def edit_tensors(self, new_batch_indices: list[int]) -> bool:
        if not defs.PRUNE_INPUTS:
            print("[Error]: edit_tensors() called from wrong simulation parameters (must use defs.PRUNE_INPUTS=True)")
            exit(0)

        # early‑out if unchanged
        # all child nodes sharing the same parent have the exact same input indices so this condition will happen often
        #if new_batch_indices == BaseModel.input_batch_indices: # commented out because now i check if the tree level changed before calling edit_tensors
            #return
       
        if not new_batch_indices:
            return

        BaseModel.input_batch_indices = new_batch_indices[:]

        #new_batch_indices = set(new_batch_indices) # if there are repeated inputs in the batch, this will return a single input (it removes duplicates...)
        # keeps only the positions (in input_batch_indices_full) that are also in new_batch_indices

        # input_batch_indices_full = [12, 323, 44, 434 , 33, 3423]
        # new_batch_indices = [12, 44, 33] -> idx = [0, 2, 4]
        idx = [i for i, val in enumerate(BaseModel.input_batch_indices_full) if val in new_batch_indices]

        # read the slices from the full (root) dataset
        # slice once; cloning is unnecessary unless you mutate in‑place
        BaseModel.input_batch = BaseModel.input_batch_full[idx]
        BaseModel.ground_truth_labels = BaseModel.ground_truth_labels_full[idx]
        self.predicted_labels  = BaseModel.top1_labels_full[idx]
        self.output_logits = BaseModel.top_5_logits_full[idx]
        self.top5_classes_indices = BaseModel.top5_classes_indices_full[idx]
        self.top5_classes_values  = BaseModel.top5_classes_values_full[idx]

        BaseModel.input_batch_size = len(BaseModel.input_batch_indices)

        # count mis‑predictions (assumes tensors)
        count_gold_top_1_mispredicted = torch.sum(self.predicted_labels != BaseModel.ground_truth_labels).item()

        # this is going to be the accuracy of another input batch, so it cannot be compared to the accuracy of the root node...
        self.batch_top1_accuracy = (BaseModel.input_batch_size - count_gold_top_1_mispredicted)/BaseModel.input_batch_size

        #custom_conv.input_ids = new_batch_indices[:]  # !!!
        # empties the tensor LUTs - must call this when running multiple inputs in the same campaign to force the LUT tensors to be loaded again to the new input
        #tcache.clear_luts()


    def layer_names(self): # Returns a list of layer names.
        return [name for name, _ in self.model.named_modules()]


    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)

        """
        if self.is_golden:
            return fw_ResNet18.forward_golden(self, *inputs)
        return fw_ResNet18.forward_faulty(self, *inputs)
        """