import torch
import atexit
import copy
import os
import json

from collections import defaultdict, deque
from typing import List, Optional
from src.models.base_model import BaseModel
from src.experiment import logger
from src import definitions as defs
from src.utils import utils as u
from src.gemmini import gemmini_config as conf
from src.flist import fi_target as fit
from src.flist.fl import fl

if defs.FI_GEMM:
    from src.flist.fault_list import FaultStatus


class Experiment():
    def __init__(self, model_faulty, model_golden):
        self.model_faulty = model_faulty
        self.model_golden = model_golden

        self.model_faulty.model.to(defs.DEVICE)
        self.model_golden.model.to(defs.DEVICE)

        # setup the fault targets
        self.fault_target = fit.setup_target()

        self.init_loggers()

        atexit.register(self._cleanup_) 


    # run_experiment: runs a fault injection campaign in one of two possible ways: 
    # 1. sequentially in the fault list, one fault per trial, or
    # 2. by sweeping the fault tree
    # this is abstract. has to be implemented in the subclass
    def run_experiment(
        self, 
        input_indices:List[int],
        trials=defs.INJECTIONS):
        raise "Error: the class Experiment should not be instantiated"
   

    # this is abstract. has to be implemented in the subclass
    def run_single_batch(self, batch_id:int, batch_indices:List[int], trials=defs.INJECTIONS):
        raise "Error: the class Experiment should not be instantiated"


    def init_loggers(self):
        if logger.SKIP_LOGS:
            print(f"{u.Co['fg'][13]}⚠️   WARNING: Logger is off!{u.R}")

        # the csv report file are stored according to this path pattern
        # the directory is build in the params.sh script file: build_dir "$root_report_path/$root_experiment/$mode_folder/trace/$model"
        # EXP_FOLDER comes from -$base_dir (the args -o $base_dir)
        batch_log_type = logger.TYPE_STATS_PER_BATCH_SEQUENTIAL

        # attach this to the report file name: if using rtl, attach the gemmini config. if SW, just attach SW
        injConf = conf.CONFIG_KEY if defs.FI_GEMM else "SW"

        if defs.TREE_FI_MODE:
            batch_log_type = logger.TYPE_STATS_PER_BATCH_PARALLEL
            fn_nodes = f"{defs.EXP_FOLDER}/nodes/{defs.CAMP_ALIAS}-s{defs.SEED}-{injConf}.csv"
            
            self.nodes_logger = logger.Logger(
                fn_nodes, 
                logger.TYPE_STATS_PER_NODE,
                skip_log=logger.SKIP_NODES_LOG) 

            fn_tree_dse = f"{defs.EXP_FOLDER}/tree_dse/{defs.CAMP_ALIAS}-s{defs.SEED}-{injConf}.csv"
            os.makedirs(fn_tree_dse, exist_ok=True)

            self.tree_dse_logger = logger.Logger(
                fn_tree_dse, 
                logger.TYPE_TREE_DSE, 
                skip_log=logger.SKEE_TREE_DSE_LOG)

        trace_path = f"{defs.EXP_FOLDER}/trace"
        batch_path = f"{defs.EXP_FOLDER}/batch"

        fn_trace = f"{trace_path}/{defs.CAMP_ALIAS}-s{defs.SEED}-{injConf}.csv"
        fn_batch = f"{batch_path}/{defs.CAMP_ALIAS}-s{defs.SEED}-{injConf}.csv"

        os.makedirs(trace_path, exist_ok=True)
        os.makedirs(batch_path, exist_ok=True)

        self.trace_logger = logger.Logger(
            fn_trace, 
            logger.TYPE_STATS_PER_FAULT if defs.FI_GEMM else logger.TYPE_STATS_PER_FAULT_SW, 
            skip_log=logger.SKIP_TRACE_LOG)

        self.batch_logger = logger.Logger(
            fn_batch, 
            batch_log_type,
            skip_log=logger.SKIP_BATCH_LOG) 


    def _cleanup_(self): # called automatically at program exit
        # dumps the remaining of the buffers if available
        
        if not logger.SKIP_LOGS:
            self.trace_logger.flush()
            self.batch_logger.flush()
     
            #if defs.TREE_FI_MODE: # removed for open source
                #self.nodes_logger.flush()


    def run_all_batches(self, input_indices:List[int], trials=defs.INJECTIONS):
        critical_faults, total_inputs = 0, len(input_indices)

        batch_id = 0

        for input_start in range(0, total_inputs, defs.BATCH_SIZE):
            ulim = min(total_inputs, input_start+defs.BATCH_SIZE)
            batch_indices = input_indices[input_start:ulim]
            
            critical_faults += self.run_single_batch(batch_id, batch_indices, trials=trials)
            batch_id += 1

        return critical_faults


    def run_single_fault(self, fault:fl.Fault):
        # sets the global fault (will be injected in matmul_random_tile())
        fl.next_fault = fault
        fl.fault_list = deque([fault])

        self.model_faulty.run_batch_inference()

        return self.eval_trial_stats()


    def run_parallel_faults(self, faults:deque[fl.Fault], is_leaf_node=False):
        # sets the global fault list as the random one
        fl.fault_list = deque(list(faults)[:])
        #fl.fault_list = copy.copy(faults)

        # marks which inputs in the batch are top1 mispredicted (maybe due to the fault or model misclassification)
        is_input_mispredicted = defaultdict(bool) 
        
        # marks which inputs in the batch are different from the golden (golden maybe ground-truth correct or not) 
        is_input_sdc1_critical = defaultdict(bool)

        # the score variation for each input index
        #score_variation = defaultdict(bool)
        #score_variation = defaultdict(float)

        # the accuracy drop status of the fault in the input
        #status_input_acc_drop = defaultdict(logger.AccDropStatus)  

        # runs every fault in the fault list at the same time, except for the tiles with collisions
        # the collision fault tiles are added back to fl.fault_list each inference
        # we have to re-run such faults until the fault list is exhausted 
        list_len = len(fl.fault_list)

        #has_any_index_var, has_any_score_var = False, False
        #has_rank_var = False

        """
        This will inject all possible faults in the list in a single forward pass. 
        Faults hitting a tile already injected previously will be skept and added back to the list. 
        If any faults were not injected, we will run this loop again until the list is exhausted
        """
        while list_len:
            self.model_faulty.run_batch_inference() 

            a, b = self.eval_trial_stats(dump_stats=is_leaf_node) # if running the tree mode: only dumps results for leaf nodes

            for index in BaseModel.input_batch_indices:
                # if any fault collisions happen, we broke a fault node into multiple fault injections - separating the collided faults from each other
                # these multiple fault injections should account for at most one critical fault per input per node, because we're counting faulty nodes
                # flags if this input was mispredicted
                is_input_mispredicted[index] |= a[index]
                is_input_sdc1_critical[index] |= b[index]  # if the inputs is critical for any faults in the node, the node is flagged as critical
            list_len = len(fl.fault_list) # this fault list is exhausted in the custom_conv implementation


        return is_input_mispredicted, is_input_sdc1_critical #, status_input_acc_drop, score_variation, has_rank_var


    # computes and logs the fi stats
    def eval_trial_stats(self, dump_stats=True, fake_masked=False): 
        # parallel injection: 
        #   - we don't dump data for non-leaf nodes (pass dump_stats=False)
        #   - for non-visited leaf nodes ONLY, we pass fake_masked=True

        # the stats of the last trial. one item for each input in the batch
        stats_list = [logger.StatsPerFault() for _ in range(BaseModel.input_batch_size_full)]

        # marks which inputs in the batch are top1 mispredicted (maybe due to the fault or model misclassification)
        is_input_mispredicted = defaultdict(bool) 
        
        # marks which inputs in the batch are different from the golden (golden maybe ground-truth correct or not) 
        is_input_sdc1_critical = defaultdict(bool)
        #is_input_sdc5_critical = defaultdict(bool)

        # the accuracy drop status of the fault in the input
        status_input_acc_drop = defaultdict(logger.AccDropStatus)  

        for i, index in enumerate(BaseModel.input_batch_indices):  # iterates over each index (index is the dataset input index)          
            lbl_grth = BaseModel.ground_truth_labels[i]
            lbl_gold = self.model_golden.predicted_labels[i]

            # parallel mode: in case a leaf-node is not visited (fake_masked WILL be True)
            # in this case, we cannot compare the work labels with the golden one
            # because the faulty model did not even run the injection on the current inputs
            # use the 'fake_masked' trick to force this case as a masked fault
            if fake_masked: # all non-visited leaf nodes have the fake_masked flag forced to True in experiment_parallel.py
                lbl_work = lbl_gold
                has_sdc1 = False
                has_sdc5 = False
            else:
                lbl_work = self.model_faulty.predicted_labels[i]
                has_sdc5 = not lbl_work in self.model_golden.top5_classes_indices[i] 
                has_sdc1 = (lbl_work != lbl_gold).item() # is there a mismatch w.r.t the gold label (maybe ground truth or not)

            mismatch_wrt_grth = (lbl_work != lbl_grth).item()  # is there a mismatch w.r.t the ground truth label (a.k.a, critical fault)
            is_input_mispredicted[index]  = mismatch_wrt_grth  # the same as sdc1
            is_input_sdc1_critical[index] = has_sdc1 
            #is_input_sdc5_critical[index] = has_sdc5
            
            """
            We define a fault as 'critical' if the top1 label != top1 golden label

            A critical fault can:
                1. drop the accuracy        (if golden == gt)
                2. improve the accuracy     (if work = gt)
                3. have no accuracy effects (if work != gt)

            input  fault  ground_truth_label  gold_label  work_label
            12      1      10                 10          10     # this fault is not critical. it causes no accuracy drops

            10      2      42                 34          56     # this fault is critical. no accuracy drops (w.r.t golden mode)
            12*     3      10                 10          12*    # this fault is critical. it causes accuracy drops, because golden == ground thruth (*)
            14      4      44*                55          44*    # this fault is critical. it improves accuracy, because the faults causes the model to classify correctly (*)
            """

            """
            # computs the accuracy drops
            if mismatch_wrt_gold:
                status_input_acc_drop[index].did_acc_drop = lbl_gold == lbl_grth
                status_input_acc_drop[index].did_acc_impr = lbl_work == lbl_grth
                status_input_acc_drop[index].did_acc_same = lbl_work != lbl_grth
            else:
                status_input_acc_drop[index].did_acc_drop = False
                status_input_acc_drop[index].did_acc_impr = False
                status_input_acc_drop[index].did_acc_same = True
            """

            # tree mode: non-leaf nodes do not log anything. dump_stats is False
            if not (dump_stats and not self.trace_logger.skip_log):
                continue

            stats_list[i].input_id = index
            stats_list[i].tgt_layer = defs.TARGET_LAYER
            stats_list[i].sdc1 = has_sdc1
            stats_list[i].sdc5 = has_sdc5

            if defs.TREE_FI_MODE:
                stats_list[i].tree_id = self.tree_id
                stats_list[i].tree_k  = self.tree_k
                stats_list[i].tree_h  = self.tree_h
                stats_list[i].th_conf_score_gap = self.th_conf_score_gap
        
            if defs.FI_GEMM:
              # the (global) last fault (fl.next_faul, Fault() object) was attributed in the last fault injection trial. 
              # here we update the status of such fault
                if fake_masked: # if this is a non-visited leaf node, we force this as a masked fault
                    self.model_faulty.stats_gemm_msk.fault_msk = [True for _ in range(BaseModel.input_batch_size_full)] # mark non-visited leaf nodes as masked faults

                else:
                    fl.next_fault.status = FaultStatus(
                        msk_gemm=self.model_faulty.stats_gemm_msk.gemm_msk[i], 
                        msk_scale=self.model_faulty.stats_gemm_msk.scale_msk[i], 
                        msk_round=self.model_faulty.stats_gemm_msk.round_msk[i], 
                        msk_clamp=self.model_faulty.stats_gemm_msk.clamp_msk[i],
                        msk_qtz=self.model_faulty.stats_gemm_msk.qtz_msk[i], 
                        critical=has_sdc1
                    )
            # copy the status from fl.next_fault to the status to be logged
            # it must be a copy because fl.next_fault will change for each iteration of the loop, 
            # and we do not want to change the status of the previous iterations on stats_list[i]
            stats_list[i].fault = copy.copy(fl.next_fault)

            self.trace_logger.try_dump_item(stats_list[i])
        
        # end of block [for i, index in enumerate(BaseModel.input_batch_indices):]

        # deals with pruned inputs of non-visited leaf nodes. we must log dummy data for these inputs
        # when using input pruning, we have to padd the trace with dummy fault injections in which the fault was "injected" 
        # but was not critical for the pruned inputs
        if defs.PRUNE_INPUTS and dump_stats and not self.trace_logger.skip_log: # if running the tree mode, this is only evaluated for leaves
            if defs.FI_GEMM:
                fl.next_fault.status = FaultStatus(
                    msk_gemm=False,  # not an actual valid result, as this was pruned and never injected
                    msk_scale=False, # not valid too
                    msk_round=False, # not valid too
                    msk_clamp=False, # not valid too
                    msk_qtz=False,   # not valid too
                    critical=False   # this is the valid result that matter. we assume it's a masked fault, but we don't know in which phase
                ) 
            
            dummy_f = fl.next_fault

            i = len(BaseModel.input_batch_indices) # the end idx of stats_list (in the previous loop)

            # logs "masked" dummy faults for the pruned inputs
            for idx_in_full in BaseModel.input_batch_indices_full: 
                if idx_in_full not in BaseModel.input_batch_indices: # if the input was pruned, we log a dummy row
                    stats_list[i].input_id = idx_in_full
                    stats_list[i].tgt_layer = defs.TARGET_LAYER
             
                    stats_list[i].fault = dummy_f
                    stats_list[i].sdc1 = False  # this is the valid results that matter. we assume it's a masked fault
                    stats_list[i].sdc5 = False  # valid result too

                    if defs.TREE_FI_MODE:
                        stats_list[i].tree_id = self.tree_id
                        stats_list[i].tree_k  = self.tree_k
                        stats_list[i].tree_h  = self.tree_h
                        stats_list[i].th_conf_score_gap = self.th_conf_score_gap

                    self.trace_logger.try_dump_item(stats_list[i])
                    i += 1
  
        return is_input_mispredicted, is_input_sdc1_critical










   