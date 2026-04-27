import csv
import os

import fault_list as fl 
from typing import Optional
from dataclasses import dataclass
from datetime import datetime


# the stats for each fault injection trial (for each individual input)
@dataclass
class StatsPerFaultGemmini:
    fault: Optional[fl.Fault]=None

    header = [
        'fault_tag', 
        # the fault layer and fault tile positions (of the unfolded weight and activation matrices)
        'tile_row', 'tile_col',
        # the gemmine fault positions
        'target', 'pe_row', 'pe_col', 'bit', 
        
        # flags for the fault outcomes in the injection layer
        'gemm_msk', 'scale_msk', 'round_msk', 'clamp_msk',        

        # flags for the fault outcomes in the end layer
        'sdc',        # the top1 predicted work label differs from the golden label
        'corrupted_elements',
    ]


TYPE_STATS_PER_FAULT_GEMMINI = 0
TYPE_STATS_PER_FAULT_SOFTWARE = 1

HEADERS = [ 
    StatsPerFaultGemmini.header,
]

MAX_BUFF_SIZE = 100 # the StatsPerFaultGemmini log buffers are flushed everytime they reach this size 

class Logger:
    def __init__(self, output_fn, log_type=TYPE_STATS_PER_FAULT_GEMMINI, skip_log=False):
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
                    writer = csv.writer(file, delimiter='\t')
                    writer.writerow(self.header)

    # buffers a new log item. if the buffer is full, dump it to the csv file
    def try_dump_item(self, new_item):
        if self.skip_log: return 
        
        self.buffer.append(new_item)

        if len(self.buffer) >= MAX_BUFF_SIZE:
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

            if self.log_type == TYPE_STATS_PER_FAULT_GEMMINI:
                for stats in self.buffer:
                    full_row = [ 
                        stats.fault.tag,
                        stats.fault.tile.a_row,
                        stats.fault.tile.b_col,
                        stats.fault.gemm.target,
                        stats.fault.gemm.pe_row,
                        stats.fault.gemm.pe_col,
                        stats.fault.gemm.bit,  # TODO: log also the cell for GL injections?

                        int(stats.fault.status.msk_gemm),
                        int(stats.fault.status.msk_scale),
                        int(stats.fault.status.msk_round),
                        int(stats.fault.status.msk_clamp),
                        int(stats.fault.status.critical),
                        int(stats.fault.status.corrupted_elements)
                    ]

                    writer.writerow(full_row)

            elif self.log_type == TYPE_STATS_PER_FAULT_SOFTWARE:
                for stats in self.buffer:
                    full_row = [ 
                        stats.fault.tag,
                        stats.fault.x,
                        stats.fault.y,
                        stats.fault.bit,
                        int(stats.fault.status.msk_gemm),
                        int(stats.fault.status.msk_scale),
                        int(stats.fault.status.msk_round),
                        int(stats.fault.status.msk_clamp),
                        int(stats.fault.status.critical),

                        int(stats.fault.status.corrupted_elements)
                    ]

                    writer.writerow(full_row)
            else:
                raise ValueError("Invalid fault type")

        self.buffer.clear()
        # clears the buffer after dumping it to the file
        self.buffer.clear()


    def flush(self):
        if self.skip_log: return 

        if len(self.buffer):
            self.dump_to_csv()





