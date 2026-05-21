import torch
import statistics

from tqdm import tqdm
from collections import defaultdict, deque
from typing import List, Optional

from src.experiment import logger
from src import definitions as defs
from src.utils import utils as u
from src.experiment import experiment as exp
from src.flist import fi_target as fit
from src.flist.fl import fl
from src.models.base_model import BaseModel

#from src import tensor_metrics as tm
#from anytree import Node, NodeMixin, RenderTree

if defs.VIT:
    from src.ivit.models.model_utils import freeze_model, unfreeze_model

class ExperimentSequential(exp.Experiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        
    # run_experiment: runs a fault injection campaign in one of two possible ways: 
    # 1. sequentially in the fault list, one fault per trial, or
    # 2. by sweeping the fault tree
    def run_experiment(self, dataset_indices:List[int], trials=defs.INJECTIONS):       
       
        # use this loop to iterate over all layers. set REPL_ALL=True to check all layers
        #for l in range(0, self.model_faulty.count_replaced_conv_layers):   #[COLLECT LAYER SHAPES]
        #    self.model_faulty.target_layer = l        #[COLLECT LAYER SHAPES]
        #    self.model_faulty.cconv.prepared = False  #[COLLECT LAYER SHAPES]

        critical_faults = self.run_all_batches(dataset_indices, trials=trials)
        return critical_faults


    def run_single_batch(self, batch_id:int, batch_indices:List[int], trials=defs.INJECTIONS):
        #print(f"Running batch {batch_id}")
        
        # for the ViT models (I-VIT) we must run the inputs batch once (to copute the max/min and scale), and then freeze such parameters for fault injection
        if defs.VIT:
            self.model_golden.calibrate = True # this flag is to allow runnnig the model with the same input again (see BaseModel "if not self.calibrate")
            self.model_faulty.calibrate = True
                        
            unfreeze_model(self.model_golden.model)
            unfreeze_model(self.model_faulty.model)

            self.model_golden.run_golden_batch(batch_indices)
            self.model_faulty.run_golden_batch(batch_indices)
            
            freeze_model(self.model_golden.model)
            freeze_model(self.model_faulty.model)

            self.model_golden.calibrate = False
            self.model_faulty.calibrate = False
        
        else:
            """
            print("Running new batch. Indices:")
            for i in batch_indices:
                print(f"{i},", end="")
            print()
            """

            self.model_golden.run_golden_batch(batch_indices) # maybe skip sub-batches with 0% model accuracy...

            if defs.FI_GEMM: # the input_ids is only used for Gemmini injection to access the tensor LUTs (CNNs only)
                self.model_faulty.gemm_conv.input_ids = batch_indices[:]

        # [COLLECT LAYER SHAPES]
        #self.model_faulty.run_batch_inference(); critical_faults = 0

        critical_fault_list, critical_faults = self.run_batch_sequential_fault_list(batch_id, trials=defs.INJECTIONS)
        #critical_fault_list, critical_faults = self.run_batch_sequential_fault_list_debug(batch_id, trials=defs.INJECTIONS)

        print(f"Finished batch {batch_id} critical faults: {critical_faults}")

        return critical_faults


    # runs fault injection trials sequentially
    def run_batch_sequential_fault_list(self, batch_id:int, trials=defs.INJECTIONS):
        # loads the fault list rows [0, trials-1]

        base_fault_list = fl.load_fault_list(
            defs.FAULT_LIST, (0, trials-1), 
            filters=self.fault_target, 
            shuffle_list=False) 

        # the list of detected critical faults
        critical_fault_list = []

        # counts the number of critical faults per each input
        count_input_critical = defaultdict(int)

        # flags, for each input, if the last fi is critical
        is_input_mispredicted = defaultdict(int)

        # the total number of critical faults after sweeping the whole base_fault_list list. a fault is critical if it causes any inputs to have a top-1 prediction different from the golden prediction 
        critical_faults = 0

        # stores the batch accuracy drop for each fi iteration
        list_batch_acc_drop = []

        # list of work accuracies (with fi)
        list_batch_work_acc = []

        # measures the average injecton time to run the fault list
        timer = u.Timer()
        timer.tic()

        total_faults = len(base_fault_list)

        for i in tqdm(range(0, total_faults), desc="Processing fault list", unit="iter", ncols=0):
            fault = base_fault_list[i] 

            is_input_mispredicted, is_input_critical = self.run_single_fault(fault)
            
            # the number of mispredicted inputs in the batch (used to compute the accuracy)
            n_mispredicted_inputs = sum(is_input_mispredicted.values())

            # flags the fault as critical if it affected any input (this cannot be used to compute the AVF directly!)
            fault_is_critical = any(is_input_critical.values())

            critical_faults += fault_is_critical

            # computes the batch accuracy for this fault node
            work_batch_accuracy = (BaseModel.input_batch_size - n_mispredicted_inputs)/BaseModel.input_batch_size

            # computes the accuracy drop w.r.t the golden accuracy
            batch_accuracy_drop = (self.model_golden.batch_top1_accuracy - work_batch_accuracy)

            list_batch_work_acc.append(work_batch_accuracy)
            list_batch_acc_drop.append(batch_accuracy_drop)

            if fault_is_critical: 
                critical_fault_list.append(fault)

        timer.toc()
        
        avg_batch_work_acc = statistics.mean(list_batch_work_acc) if len(list_batch_work_acc) else 0
        avg_batch_acc_drop = statistics.mean(list_batch_acc_drop) if len(list_batch_acc_drop) else 0

        self.batch_logger.dump_item(
            logger.StatsPerBatchSequential(
                batch_id, 
                defs.TARGET_LAYER, 
                critical_faults, 
                self.model_golden.batch_top1_accuracy, 
                avg_batch_work_acc, 
                avg_batch_acc_drop, 
                timer.time_measure
                )
            )

        return critical_fault_list, critical_faults


    def run_batch_sequential_fault_list_debug(self, batch_id:int, trials=defs.INJECTIONS):
        #import src.utils.tensor_metrics as tm
        import matplotlib.pyplot as plt
        import time

        # loads the fault list rows [0, trials-1]
        base_fault_list = fl.load_fault_list(
            defs.FAULT_LIST, 
            (0, trials-1), 
            filters=self.fault_target, 
            shuffle_list=False
            ) 

        print(f"Loaded fault list of size {len(base_fault_list)}")

        total_faults = len(base_fault_list)

        plt.ion()  # turn on interactive mode
        fig, ax = plt.subplots()
        xs, ys = [], []
        line, = ax.plot(xs, ys)
    
        start = time.time()
        interval=0.4

        fiMode = (
            "Gates" if defs.FI_GEMM and defs.ENABLE_GL_FAULT_MODEL
            else "RTL" if defs.FI_GEMM
            else "Software"
        )

        # Create the text object only ONCE
        text_handle = ax.text(
            0, 1.03,  # position in axes coordinates
            "",       # initial text
            fontsize=14,
            transform=ax.transAxes
        )

        ax.set_title(f"Injection mode: {fiMode}", fontsize=14)
        ax.set_ylabel("Logits noise (%)", fontsize=14)
        ax.set_xlabel("Fault iteraction", fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        critical_faults, ninjected = 0, 0
        samples = []
           
        for i in tqdm(range(0, total_faults), desc="Processing fault list", unit="iter", ncols=0):
        #for fault in base_fault_list:
            fault = base_fault_list[i] # ATTENTION: uncomment this if using tqdm

            is_input_mispredicted, is_input_critical = self.run_single_fault(fault)
            ninjected += 1

            n_mispredicted_inputs = sum(is_input_mispredicted.values())

            # flags the fault as critical if it affected any input (this cannot be used to compute the AVF directly!)
            fault_is_critical = any(is_input_critical.values())

            critical_faults += fault_is_critical

            # computes the batch accuracy for this fault node
            work_batch_accuracy = (BaseModel.input_batch_size - n_mispredicted_inputs)/BaseModel.input_batch_size

            # computes the accuracy drop w.r.t the golden accuracy
            #batch_accuracy_drop = 100*(self.model_golden.batch_top1_accuracy - work_batch_accuracy)
            
            noise = (self.model_faulty.output_logits - self.model_golden.output_logits)/ \
                torch.mean(self.model_golden.output_logits).item()

            noise = 100*torch.mean(noise).item()
    
            #psrn = tm.compute_psnr(self.model_faulty.output_logits, self.model_golden.output_logits)
            #samples.append(batch_accuracy_drop)

            samples.append(noise)
            mean_noise = statistics.mean(samples)
            #ys.append(batch_accuracy_drop)
            ys.append(noise) 
            #xs.append(time.time() - start)
            xs.append(ninjected) # time.time() - start)

            if fault_is_critical:
                ax.scatter(xs[-1], ys[-1], color="red", s=20)

            line.set_xdata(xs)
            line.set_ydata(ys)

            ax.relim()           # recompute limits
            ax.autoscale_view()  # rescale axes
            
            """
            text_handle.set_text(
                f"- Critical faults: {critical_faults}/{ninjected} ({100*critical_faults/total_faults:.1f}%)\n" \
                f"- PSNR (last): {psrn:.2f}\n" \
                f"- PSNR (mean): {mean:.2f}"
            )
            """
            text_handle.set_text(
                f"- Total injected:  {ninjected}/{total_faults}\n"
                f"- Critical faults: {critical_faults}/{ninjected} ({100*critical_faults/ninjected:.1f}%)\n" \
                f"- Mean noise:      {mean_noise:.1f}%\n" \
            )

            plt.draw()
            plt.pause(0.004)     # required for GUI update
            time.sleep(interval)

        plt.ioff()         # turn OFF interactive mode
        plt.show()         # blocks until you close the window

        return [], critical_faults