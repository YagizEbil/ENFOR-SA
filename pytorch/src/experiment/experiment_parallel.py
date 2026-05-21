import torch
import statistics
import time
import random
import copy
import math

from tqdm import tqdm
from collections import defaultdict, deque
from typing import List, Optional
from anytree import Node, NodeMixin, RenderTree

from src.models.base_model import BaseModel
from src.experiment import logger as logger
from src import definitions as defs
from src.utils import utils as u
from src.experiment import experiment as exp
from src.flist import fi_target as fit
from src.flist import fifo_queue as fifo
from src.flist.fl import fl
from src.flist import tree as tree

if defs.VIT:
    from src.ivit.models.model_utils import freeze_model, unfreeze_model


def compute_p(margin):
    """
        p^(x)∝Pr(fault changes argmax)≈f(margin(x))

        p^(x)≈exp(−α⋅m(x))

        Computes an artificial failure probability (p) based on the top1-top2 confidence gap (margin)
        p should decay exponentially as the gap increases (according to the procedure shown by gpt)
        increasign the alpha factor makes the decay more aggressive, meaning p decreases quicker for larger alpha values
        reduced p values (with higher alpha), produces trees with higher dept, so
        the time to build the tree also grows very fast for larger values of alpha 
    """
    alpha = 1.5
    return math.exp(-alpha*margin)

    
class ExperimentParallel(exp.Experiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


        # keeps track if each batch was already run in golden mode
        # this is because in the tree mode, multiple trees can exist per batch, calling run_single_batch() once per batch per tree
        # we must run each golden batch only once

        self.was_gold_batch_run = [False for _ in range(defs.BATCHES)]

    # run_experiment: runs a fault injection campaign in one of two possible ways: 
    # 1. sequentially in the fault list, one fault per trial, or
    # 2. by sweeping the fault tree
    def run_experiment(self, dataset_indices:List[int], trials=defs.INJECTIONS):
        USE_FIXED_TREE_PARAMS = True

        if USE_FIXED_TREE_PARAMS:
            K_fixed = 25
            H_fixed = 4

            K_min, K_max, K_step = K_fixed, K_fixed+1, 1
            H_min, H_max, H_step = H_fixed, H_fixed+1, 1
        
        else:
            # the 'k' number of child nodes in the tree
            K_min, K_max, K_step = 2, 8, 1
            # the tree height
            H_min, H_max, H_step = 2, 8, 1  # the max height should be 7. otherwise I run out of memory
            # the conf. thresholds
            Th_min, Th_max, Th_step = 0, 51, 10


        Th_fixed = 20 # inputs with top1-top2 confg gap <= Th are kept in the batch, even if golden
        #Th_min, Th_max, Th_step = Th_fixed, Th_fixed+1, 1
        self.th_conf_score_gap = Th_fixed/100.0

        critical_faults = 0

        self.base_fault_list = fl.load_fault_list(
            defs.FAULT_LIST, 
            (0, trials-1), 
            filters=self.fault_target, 
            shuffle_list=False
            )

        """
        i'll start by choosing k = 3, and then computing h based on an p estimation
        lower p:  requires higher h -> prunes more nodes with less evaluations
        higher p: use lower h -> there as a high probability of having to inspect the leaf nodes anyways. the sooner we get to the leaves, the better
        
    
        Simplify as follows:
            1. take all confident inputs, with no CF
            2. find the best tree for those inputs with a small sample of FI. use a single tree config for this case

            3. for the non-confident inputs:
                optimize the tree based on the p estimations
        """

        def compute_p(conf_gap):
            alpha = 2
            return math.exp(-alpha*conf_gap)

        for k in range(K_min, K_max, K_step):
            for h in range(H_min, H_max, H_step):
                print(f"Running tree with k={k}, h={h}")
               
                self.tree_k = k
                self.tree_h = h

                forest = tree.build_forest(k, h, trials)
                #tree.print_forest(forest)

                self.forest_size = len(forest)

                # [Tree info]
                #print(f"Building trees with (k, h) = ({k}, {h})") #, end="")
                print(f"Forest of size is {len(forest)}")

                critical_faults = 0 # this must be reset before running each forest

                #forest_timer = u.Timer(); forest_timer.tic()

                for self.tree_id, root in enumerate(forest):
                    # [Tree info]
                    print(f"* Running tree {root.name} - Faults ∈ [{root.interval[0]},{root.interval[1]}]:")

                    """
                    For each root node (for each tree):
                        - Load the full fault list to the root node
                        - Each child node will then load a sublist from self.base_fault_list. each node has it's own list indexes (node.interval) 
                    """
                    # assigns the fault list to the root
                    self.current_root = root
                    critical_faults += self.run_all_batches(dataset_indices, trials)

                #forest_timer.toc()

                #tree.print_forest(forest)
                print(f"Finished forest with critical faults: {critical_faults}")

        return critical_faults


    def run_single_batch(self, batch_id:int, batch_indices:List[int], trials=defs.INJECTIONS):
        # [Tree info]
        print(f"  > Running batch {batch_id}")

        # for the ViT models (I-VIT) we must run the input batch once (to copute the max/min and scale), and then freeze such parameters
        if defs.VIT:
            if not self.was_gold_batch_run[batch_id]:
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

                self.was_gold_batch_run[batch_id] = True
        else:
            if not self.was_gold_batch_run[batch_id]:
                self.model_golden.run_golden_batch(batch_indices) # maybe skip sub-batches with 0% model accuracy...
                self.was_gold_batch_run[batch_id] = True

            if defs.FI_GEMM: # the input_ids is only used for Gemmini injection to access the tensor LUTs (CNNs only)
                self.model_faulty.gemm_conv.input_ids = batch_indices[:]

        critical_fault_list, critical_faults = self.run_batch_tree_fault_list(batch_id, trials=trials)

        #critical_faults = self.DEBUG_run_batch_tree_fault_list(batch_id, trials=trials)

        return critical_faults


    # runs the fault list in the tree mode
    def run_batch_tree_fault_list(self, batch_id:int, trials=defs.INJECTIONS):

        # initializes the root input indexes containing all inputs in the batch
        # i'm doing this here because it must be done after run_golden_batch() 
        self.current_root.input_indices = BaseModel.input_batch_indices[:]

        # the tree is shared across all batches, so we must set each node as non-visited before each new batch run
        for pre, _, node in RenderTree(self.current_root):
            node.visited = False
            node.is_critical = False

        # marks the level of the last visited node
        self.last_visited_parent = -1

        # the maximum level explored in the tree fault list across all fault injections
        self.max_reached_level = 0

        # the number of reached failed leaves (critical faults) identified in the tree fault list
        visited_nodes, reached_leaves, critical_faults = 0, 0, 0

        # stores the batch accuracy drop for each fi iteration
        list_batch_acc_drop = []

        # stores the average accuracy drop for each number of parallel faults
        acc_drop: defaultdict[int, list[float]] = defaultdict(list)

        # counts the number of nodes with accuracy drop > 0, < 0 or = 0
        #count_positive_acc_drop, count_negative_acc_drop, count_zero_acc_drop = 0, 0, 0

        # counts the number visited non-leaf nodes
        count_visited_non_leaf_nodes = 0

        # list of work accuracies (with fi)
        list_batch_work_acc = []

        # counts the number of critical faults per each input. accumulates over all injected faults
        count_input_critical = defaultdict(int)

        # the queue containing the fault nodes from the tree
        fault_queue = fifo.FIFOQueue()

        # start with the root node. child nodes are added to the queue dynamically if a given node is marked as critical
        fault_queue.push(self.current_root)

        # measures the average injecton time to run the fault list
        timer = u.Timer()

        # fault list of all non-leaf nodes with zero acc drop
        #debug_fault_list = []
        #ratio_pruned = []

        # the list of detected critical faults
        critical_fault_list = []

        timer.tic()

        # this was set to true for all non-visited leaf nodes to log dummy faults. for sw fi we must set this back to false manually

        if defs.FI_GEMM:
            self.model_faulty.stats_gemm_msk.fault_msk = [False for _ in range(BaseModel.input_batch_size_full)] 

        # iterates over the fault queue (the tree is explored in BFS) 
        # if a given node is marked as critical, its children are added to the queue dynamically
        while not fault_queue.is_empty():
            fault_node = fault_queue.pop()

            # the last nodes can be empty if there's not enough faults to fill the tree
            # these are just skept from simulation
            if fault_node.interval == None:
                continue

            if defs.PRUNE_INPUTS:
                if len(fault_node.input_indices) == 0: # if this node had all its inputs pruned (in its parent node) due to masking, we just skip it
                    continue

            is_input_mispredicted, is_input_critical = self.run_fault_node(fault_node)

            #is_input_mispredicted, is_input_critical, status_input_acc_drop, score_variation, has_rank_var = self.run_fault_node(fault_node)

            # the number of mispredicted inputs in the batch (used to compute the accuracy)
            n_mispredicted_inputs = sum(is_input_mispredicted.values())

            # the number of inputs in the batch != golden (used to flag as critical fault)
            n_critical_inputs = sum(is_input_critical.values()) 

            # computes the batch accuracy for this fault node
            work_batch_accuracy = (BaseModel.input_batch_size - n_mispredicted_inputs)/BaseModel.input_batch_size
           
            # computes the accuracy drop w.r.t the golden accuracy
            batch_accuracy_drop = (self.model_golden.batch_top1_accuracy - work_batch_accuracy)

            # flags the fault as critical if it affected any input (this cannot be used to compute the AVF directly!)
            fault_node.is_critical = n_critical_inputs != 0
            
            visited_nodes += 1

            if fault_node.is_leaf: # critical node leaves are marked as critical faults
                critical_faults += fault_node.is_critical
                reached_leaves += 1
                list_batch_acc_drop.append(batch_accuracy_drop)
                list_batch_work_acc.append(work_batch_accuracy)

                #if fault_node.is_critical:
                    #critical_fault_list.append(fault_node.fault_list[0].tag)
            else:
                keep_node = False # use False for DAC
                
                # if we use input pruning, then we're using our approch. i'm using this here just to check 
                # weather the simul is for dac or us. if defs.PRUNE_INPUTS is False, then it's simulating dac 
                # and we should let keep_node=False
              
                if defs.PRUNE_INPUTS: # only using keep_node for our proposal, and not dac
                    for index in fault_node.input_indices:
                        if not is_input_critical.get(index):  # if the index is not so far considered critical
                            keep_node |= self.model_golden.conf_gap[index] < 5.0/100
                            #keep_node |= self.model_golden.conf_gap[index] < 6.0/100
                            #keep_node |= self.model_golden.conf_gap[index] < 3.0/100

                if fault_node.is_critical or keep_node or fault_node.is_root:
                    # the child node takes only the parent node's indexes which are not golden       

                    if defs.PRUNE_INPUTS:
                        #alpha, beta = 0.2, 0.8  # alpha: input score variation weight, beta: node criticality weight
                        child_indices = []
                        
                        # we only prune the inputs if its golden AND the node criticality is below a threshold
                        for index in fault_node.input_indices:
                            if is_input_critical.get(index):  # always keep non-golden inputs
                               child_indices.append(index)

                            # TODO: as we go down the three, the probability of pruning golden inputs should decrease
                            else: 
                                # if the the top1 - top2 confidence gap is less than this threshold, this input is more likely to be critical, so we do not prune it
                                keep_golden = self.model_golden.conf_gap[index] < self.th_conf_score_gap
                                #keep_golden = False
                                if keep_golden: child_indices.append(index)

                        #ratio_pruned.append(1-len(child_indices)/len(fault_node.input_indices))

                    else: # no input pruning
                        child_indices = fault_node.input_indices[:]

                    for child in fault_node.children:
                        #child.parent_score_var = sum(score_variation.values())
                        child.input_indices = child_indices[:]
                        fault_queue.push(child)
 
            """
            if not self.nodes_logger.skip_log: # Warning: some code below may be used even if the logger is off (the blocks above that are for now removed)
                fault_node.count_across_batch_visits += 1
                fault_node.count_across_batch_critical += n_critical_inputs/BaseModel.input_batch_size
                fault_node.criticality_across_batches = fault_node.count_across_batch_critical/fault_node.count_across_batch_visits
                fault_node.criticality = n_critical_inputs/BaseModel.input_batch_size

                self.nodes_logger.try_dump_item(
                    logger.StatsPerNode(
                        node_id=fault_node.name, 
                        batch_id=batch_id, 
                        visits=fault_node.count_across_batch_visits, 
                        criticality=fault_node.criticality,
                        tree_id=self.tree_id,
                        tree_k=self.tree_k,
                        tree_h=self.tree_h))
            """

        # end not fault_queue.is_empty(): 

        timer.toc()

        # Logs dummy inputs for non-visited leaf nodes
        # non-visited leaves are equivalent to an injected node with 0 critical faults
        for leaf in self.current_root.leaves:
            if leaf.interval != None: # only process this if the leaf is a valid FI
                if not leaf.visited:
                    fl.next_fault = self.base_fault_list[leaf.interval[0]] # we need the tag for eval_trial_stats()...
                    leaf.tag = fl.next_fault.tag # only to show the tag when the tree is printed
                    self.eval_trial_stats(dump_stats=True, fake_masked=True) # log the dummy stats

        avg_batch_acc_drop = statistics.mean(list_batch_acc_drop) if len(list_batch_acc_drop) else 0

        # if len(list_batch_work_acc) is zero, meaning no leaf nodes were reached due to fault masking, i'm seting work accuracy = golden accuracy
        avg_batch_work_acc = statistics.mean(list_batch_work_acc) if len(list_batch_work_acc) \
            else self.model_golden.batch_top1_accuracy

        # [Tree info]
        print(f"     + Visited nodes:   {visited_nodes}")
        print(f"     + Reached leaves:  {reached_leaves}")
        print(f"     + Critical leaves: {critical_faults}")

        self.batch_logger.dump_item(
            logger.StatsPerBatchParallel(
                batch_id=batch_id, 
                tgt_layer=defs.TARGET_LAYER, 
                failed_leaves=critical_faults, 
                reached_leaves=reached_leaves, 
                visited_nodes=visited_nodes,
                tree_id=self.tree_id,
                tree_k=self.tree_k,
                tree_h=self.tree_h,
                max_reached_level=self.max_reached_level, 
                batch_gold_accuracy=self.model_golden.batch_top1_accuracy, 
                avg_batch_work_accuracy=avg_batch_work_acc, 
                avg_batch_accuracy_drop=avg_batch_acc_drop, 
                avg_injection_time=timer.time_measure, 
                th_conf_score_gap=self.th_conf_score_gap
                )
            )

        # no need to log any StatsTreeDSE anymore. all of this info is in the log above (StatsPerBatchParallel) (appart from forest_size)
        """
        self.tree_dse_logger.dump_item( 
                logger.StatsTreeDSE(
                    batch_id=batch_id,
                    tree_id=self.tree_id,
                    tree_k=self.tree_k, 
                    tree_h=self.tree_h, 
                    trees=self.forest_size, 
                    failed_leaves=critical_faults,
                    layer=defs.TARGET_LAYER,
                    time=timer.time_measure
                    ) 
                )
        """
        
        #treeflist.draw_tree(self.current_root.root);exit(0);
        return critical_fault_list, critical_faults


    # runs a single fault_node from the tree
    def run_fault_node(self, fault_node:tree.FaultNode):
        fl.next_fault_is_leaf = fault_node.is_leaf

        parent_changed = fault_node.parent != self.last_visited_parent
        
        self.last_visited_parent = copy.copy(fault_node.parent)

        # prunes the golden inputs in the batch for this node
        # all child nodes sharing the same parent have the exact same input indices
        # we don't need to edit the tensors again if a "sibling" of this node has already been visited
        if defs.PRUNE_INPUTS and parent_changed: 
            self.model_golden.edit_tensors(fault_node.input_indices)

            if len(fault_node.input_indices) and defs.FI_GEMM and not defs.VIT: # the caching trick is not used in ViTs
                self.model_faulty.gemm_conv.input_ids = fault_node.input_indices

        self.max_reached_level = max(self.max_reached_level, fault_node.depth)

        # loads the portion of the full fault list for this node
        fault_node.fault_list = fl.load_sub_fault_list(self.base_fault_list, fault_node.interval)

        fl.next_fault = fault_node.fault_list[0] # for book keeping (e.g, if this is a leaf node, will need to log this fault's status)
        
        if fault_node.is_leaf:
            fault_node.tag = fl.next_fault.tag

        fault_node.visited = True

        return self.run_parallel_faults(
            deque(list(fault_node.fault_list)[:]), 
            is_leaf_node=fault_node.is_leaf
            )
