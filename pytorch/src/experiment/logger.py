import csv
import os
from typing import Optional
from dataclasses import dataclass
from datetime import datetime
from src import definitions as defs
from src.flist.fl import fl


# [Debug only] disable logging FI results
SKIP_LOGS = False

SKIP_TRACE_LOG = SKIP_LOGS
SKIP_BATCH_LOG = SKIP_LOGS
SKIP_NODES_LOG = True
SKEE_TREE_DSE_LOG = SKIP_LOGS


# the stats for each fault injection trial (for each individual input)
@dataclass
class StatsPerFault:
    # input id and targets
    input_id:  int=0
    tgt_layer: int=0

    # stores the fault positions and status
    fault: Optional[fl.Fault]=None

    # app-level errors (not stored in the Fault class...)
    sdc1: bool=False  # top1 label != golden label (a.k.a, a critical fault)
    sdc5: bool=False  # top1 label is none of the top5 golden labels

    # the CSV header for the data StatsPerFault
    if defs.FI_GEMM:
        header = [
            'input_id', 
            'fault_tag', 
            # the fault layer and fault tile positions (of the unfolded weight and activation matrices)
            'layer', 'tile_row', 'tile_col',
            # the gemmine fault positions
            'target', 'pe_row', 'pe_col', 'bit',        
           
            # flags for the fault outcomes in the end layer
            'sdc1',        # the top1 predicted work label differs from the golden label
            'sdc5',        # the top1 predicted work label is none of the top5 golden labels

            # flags for the fault outcomes in the injection layer
            'gemm_msk', 'scale_msk', 'round_msk', 'clamp_msk', 'qtz_msk', 
        ]

    else: # SW Injection headers
        if defs.VIT:
            header = [
                'input_id', 
                'fault_tag', 
                'layer',
                'x', 'y', 'target', 'bit',        
                # flags for the fault outcomes in the end layer
                'sdc1',        # the top1 predicted work label differs from the golden label
                'sdc5',        # the top1 predicted work label is none of the top5 golden labels
            ]

        else:
            header = [
                'input_id', 
                'fault_tag', 
                # the fault layer and fault tile positions (of the unfolded weight and activation matrices)
                'layer',
                # the gemmine fault positions
                'target', 'N', 'C', 'H', 'W', 'bit',        
                # flags for the fault outcomes in the end layer
                'sdc1',        # the top1 predicted work label differs from the golden label
                'sdc5',        # the top1 predicted work label is none of the top5 golden labels
            ]

    # [Parallel mode only]: tree parameters
    if defs.TREE_FI_MODE:
        tree_id: int=0
        tree_k: int=0
        tree_h: int=0
        th_conf_score_gap: float=0 # only prune inputs if the the top1 - top2 confidence gap is higher than this threshold    
        header.extend(['tree_id', 'k', 'h', 'conf_score_th'])


@dataclass
class StatsTreeDSE:
    batch_id:int=0
    tree_id:int=0
    tree_k: int=0  # the number of child nodes
    tree_h: int=0   # the hight
    trees: int=0 # the required number of trees to fit all injections with trees of shape (k, h)
    failed_leaves: int=0 # the total number of critical faults (sum of all failed leaves accross all trees)
    layer:int=0
    time: int=0

    header = [
        'batch_id', 
        'tree_id', 
        'k', 
        'h', 
        'trees', 
        'critical_faults',
        'layer', 
        'time'
    ]


# the batch-level stats for the parallel (tree) fault injection approach
@dataclass
class StatsPerBatchParallel:
    # input id and targets
    batch_id:  int=0  # the batch id
    tgt_layer: int=0  # target conv layer id
    failed_leaves:  int=0    # the number of critical reached leaves in the tree (a.k.a, critical_faults)
    reached_leaves: int=0    # the number of reached leaves in the tree
    visited_nodes:  int=0
    tree_id: int=0
    tree_k: int=0  # the tree parameter (number of children nodes)
    tree_h: int=0,
    max_reached_level: int=0 # the maximum level reached in the tree
    batch_gold_accuracy:     float=0  # the golden accuracy for the batch
    avg_batch_work_accuracy: float=0  # the average batch accuracy after a set of injections (avg across all injections) 
    avg_batch_accuracy_drop: float=0  # the average accuracy drop of avg_batch_work_accuracy w.r.t  batch_gold_accuracy
    avg_injection_time: float=0       # the average inj. time in seconds
    th_conf_score_gap: float=0 

    # the CSV header for the data StatsPerBatchParallel
    header = [
        'batch_id',
        'layer',   
        'critical_faults',
        'reached_leaves', 
        'visited_nodes', 
        'tree_id', 'k', 'h',
        'max_level',
        'avg_batch_gold_acc', 
        'avg_batch_work_acc', 
        'avg_batch_acc_drop',
        'injection_time',
        'conf_score_th'
    ]


# the batch-level stats for the sequential fault injection approach
@dataclass
class StatsPerBatchSequential:
    batch_id:  int=0
    tgt_layer: int=0
    critical_faults: int=0
    batch_gold_accuracy:   float=0
    avg_batch_work_accuracy: float=0
    avg_batch_accuracy_drop: float=0
    avg_injection_time: float=0
    
    # debug stuff
    #fault_coverage: float=0 # [debug only] when running a list of only critical faults, fault_coverage counts the percentage of such critical faults that are not masked due to fault interference
    #min_dist: int=1 # [debug only] when running a list of only critical faults. the minimum required "distance" between the critical fault and interference faults
    #avg_dist: float=0 # [debug only] the average distance of the random interference faults w.r.t the critical one

    # the CSV header for the data StatsPerBatchSequential
    header = [
        'batch_id',
        'layer',    
        'critical_faults', 
        'avg_batch_gold_acc', 
        'avg_batch_work_acc', 
        'avg_batch_acc_drop', 
        'injection_time'
    ]


# this tells, for each input (in llog_trial_stats) if the fault in the input will cause the accuracy to change in a given way (drop, improve or same)
@dataclass
class AccDropStatus:
    did_acc_drop: bool = False
    did_acc_impr: bool = False
    did_acc_same: bool = False


@dataclass
class StatsPerNode:
    node_id: int
    batch_id: int
    visits: int
    criticality: float

    tree_id: int=0
    tree_k:int=0
    tree_h:int=0
    header = [
        "node_id", 
        "batch_id",
        'tree_id',
        'k', 
        'h'
        "visits", 
        "criticality",  
    ]


@dataclass
class StatsInputConfidence:
    # input id and targets
    input_id:  int=0
    pred_correct: int=0
    top1_score: float=0
    top2_score: float=0

    header=[
        "input_id",
        "pred_correct",
        "top1_score",
        "top2_score"
    ]

# which type of log is this (StatsPerFault, StatsPerBatchParallel, StatsPerBatchSequential,...). used to properly dump the data to the CSV file

# logs information per each fault injection (goes to the 'trace' folder)
TYPE_STATS_PER_FAULT = 0
TYPE_STATS_PER_FAULT_SW = 1 

# logs results per each batch (goes to the 'batch' folder)
TYPE_STATS_PER_BATCH_PARALLEL   = 2
TYPE_STATS_PER_BATCH_SEQUENTIAL = 3

# logs the nodes' information for the tree injection approach
TYPE_STATS_PER_NODE  = 4

# logs for the k,h tree dse
TYPE_TREE_DSE = 5

# the PDF for the input conf. scores
TYPE_STATS_CONFIDENCE = 6

HEADERS = [ 
    StatsPerFault.header,           # TYPE_STATS_PER_FAULT
    StatsPerFault.header,           # TYPE_STATS_PER_FAULT_SW 
    StatsPerBatchParallel.header,   # TYPE_STATS_PER_BATCH_PARALLEL
    StatsPerBatchSequential.header, # TYPE_STATS_PER_BATCH_SEQUENTIAL
    StatsPerNode.header,            # TYPE_STATS_PER_NODE
    StatsTreeDSE.header,
    StatsInputConfidence.header
]

MAX_BUFF_SIZE = 100 # the StatsPerFault log buffers are flushed everytime they reach this size 

class Logger:
    def __init__(self, output_fn, log_type=TYPE_STATS_PER_FAULT, skip_log=False):
        self.header = HEADERS[log_type]
        self.output_fn = output_fn
        self.buffer = []
        self.log_type = log_type
        self.skip_log = skip_log

        # if the file does not exist, create it and write its header
        if not self.skip_log:
            if not os.path.exists(self.output_fn):
                with open(self.output_fn, mode='w', newline='') as file:
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    file.write(f"# Created on: {now}\n")
                    file.write(f"# Alias:      {defs.CAMP_ALIAS}\n")
                    file.write(f"# Batches:    {defs.BATCHES}\n")
                    file.write(f"# Batch size: {defs.BATCH_SIZE}\n")
                    file.write(f"# Seed:       {defs.SEED}\n")
                    writer = csv.writer(file, delimiter='\t')
                    writer.writerow(self.header)


    # buffers a new log item. if the buffer is full, dump it to the csv file
    def try_dump_item(self, new_item):
        if self.skip_log: return 
        
        self.buffer.append(new_item)

        if self.log_type == TYPE_STATS_PER_FAULT or \
           self.log_type == TYPE_STATS_CONFIDENCE or \
           self.log_type == TYPE_STATS_PER_NODE: 

           # not using buffer for the other log types...
            if len(self.buffer) >= MAX_BUFF_SIZE:
                self.dump_to_csv()
        else:
            self.dump_to_csv()


    # buffers the item and dumps it to the csv file
    def dump_item(self, new_item):
        if self.skip_log: return 
        self.buffer.append(new_item)
        self.dump_to_csv()


    def dump_to_csv(self):
        # dumps the buff to the CSV log file
        with open(self.output_fn, mode='a', newline='') as file:
            writer = csv.writer(file, delimiter='\t')

            if self.log_type == TYPE_STATS_CONFIDENCE:
                for stats in self.buffer:
                    writer.writerow(
                    [
                        stats.input_id,
                        stats.pred_correct,
                        f"{stats.top1_score:.4f}",
                        f"{stats.top2_score:.4f}",
                    ]
                )

            elif self.log_type == TYPE_TREE_DSE:
                for stats in self.buffer:
                      writer.writerow(
                        [
                            stats.batch_id,
                            stats.tree_id,
                            stats.tree_k,
                            stats.tree_h,
                            stats.trees, 
                            stats.failed_leaves, 
                            stats.layer,
                            f"{stats.time:.2f}"
                        ]
                    )

            elif self.log_type == TYPE_STATS_PER_FAULT: 
                for stats in self.buffer:
                    full_row = [ 
                        stats.input_id,
                        stats.fault.tag,
                        stats.tgt_layer,
                        stats.fault.tile.a_row,
                        stats.fault.tile.b_col,
                        stats.fault.gemm.target,
                        stats.fault.gemm.pe_row,
                        stats.fault.gemm.pe_col,
                        stats.fault.gemm.bit,  # TODO: log also the cell for GL injections?
                        int(stats.sdc1),
                        int(stats.sdc5),
                        int(stats.fault.status.msk_gemm),
                        int(stats.fault.status.msk_scale),
                        int(stats.fault.status.msk_round),
                        int(stats.fault.status.msk_clamp),
                        int(stats.fault.status.msk_qtz),
                    ]

                    if defs.TREE_FI_MODE:
                        full_row.extend(
                            [
                                stats.tree_id,
                                stats.tree_k,
                                stats.tree_h,
                                stats.th_conf_score_gap
                            ])

                    writer.writerow(full_row)

            elif self.log_type == TYPE_STATS_PER_FAULT_SW:
                if defs.VIT:
                    for stats in self.buffer:
                        full_row = [ 
                            stats.input_id,
                            stats.fault.tag,
                            stats.tgt_layer,
                            stats.fault.x,
                            stats.fault.y,
                            stats.fault.target,
                            stats.fault.bit,
                            int(stats.sdc1),
                            int(stats.sdc5),
                        ]

                    if defs.TREE_FI_MODE:
                        full_row.extend(
                            [
                                stats.tree_id,
                                stats.tree_k,
                                stats.tree_h,
                                stats.th_conf_score_gap
                            ])
                    writer.writerow(full_row)

                else:
                    for stats in self.buffer:
                        full_row =[ 
                            stats.input_id,
                            stats.fault.tag,
                            stats.tgt_layer,
                            stats.fault.target,
                            stats.fault.N,
                            stats.fault.C,
                            stats.fault.H,
                            stats.fault.W,
                            stats.fault.bit,
                            int(stats.sdc1),
                            int(stats.sdc5)
                        ]

                        if defs.TREE_FI_MODE:
                            full_row.extend(
                                [
                                    stats.tree_id,
                                    stats.tree_k,
                                    stats.tree_h,
                                    stats.th_conf_score_gap
                                ])

                    writer.writerow(full_row)

            elif self.log_type == TYPE_STATS_PER_BATCH_PARALLEL: 
                for stats in self.buffer: 
                    writer.writerow(
                        [
                            stats.batch_id, 
                            stats.tgt_layer, 
                            stats.failed_leaves,
                            stats.reached_leaves, 
                            stats.visited_nodes,
                            stats.tree_id,
                            stats.tree_k,
                            stats.tree_h,
                            stats.max_reached_level, 
                            f'{stats.batch_gold_accuracy:.4f}',
                            f'{stats.avg_batch_work_accuracy:.4f}',
                            f'{stats.avg_batch_accuracy_drop:.4f}',
                            f'{stats.avg_injection_time:.2f}',
                            stats.th_conf_score_gap
                        ]
                    )

            elif self.log_type == TYPE_STATS_PER_BATCH_SEQUENTIAL:
                for stats in self.buffer: 
                    writer.writerow(
                        [
                            stats.batch_id,
                            stats.tgt_layer,
                            stats.critical_faults, 
                            f'{stats.batch_gold_accuracy:.4f}',
                            f'{stats.avg_batch_work_accuracy:.4f}',
                            f'{stats.avg_batch_accuracy_drop:.4f}',
                            f'{stats.avg_injection_time:.2f}',
                        ]
                    )

            elif self.log_type == TYPE_STATS_PER_NODE: 
                for stats in self.buffer: 
                    writer.writerow(
                        [
                            stats.node_id, 
                            stats.batch_id,
                            stats.tree_id,
                            stats.tree_k,
                            stats.tree_h,
                            stats.visits,
                            f'{stats.criticality:.4f}'
                        ]
                    )
            else:
                raise "Error: invalid log type"
        
        # clears the buffer after dumping it to the file
        self.buffer.clear()


    def flush(self):
        if self.skip_log: return 

        if len(self.buffer):
            self.dump_to_csv()





