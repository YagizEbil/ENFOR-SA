#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from collections import defaultdict
from tabulate import tabulate
from scipy.stats import wasserstein_distance
from scipy.stats import ks_2samp
from scipy.special import rel_entr  # elementwise P log(P/Q)


# In[2]:


# The full reports folder path 
REPORTS_PATH="./reports"

# Parameters matching the ones in the experiment file
SEED = 0
ALIAS = "sim-xyz"
CONFIG_KEY = "OSDIM8"
IS_SW_LOG=False

# In[3]:


"""
Loads a model's report

model: the model name.
experiment: the name of the experiment
mode:
    - "sequential"
    - "parallel"
area: the folder where the reports are stored.
example, for the logs in reports/abc/ResNet18 -> area is abc

log_type:
    - "trace" for trace files (results for each injection, each input)
    - "batch" for the batch files (results for each batch, each layer)
    
"""
def load_df(
    model="ResNet18", 
    experiment="exp-v1", 
    mode="sequential", 
    area="workshop", 
    log_type="trace"):

    # the file name is structured as
    #fn = "<alias>-s<sed>-<config key>.csv"

    if IS_SW_LOG: #area == "sw":
        file = f"{ALIAS}-s{SEED}-SW.csv"
    else:
        file = f"{ALIAS}-s{SEED}-{CONFIG_KEY}.csv"
        
    fn = f"./{REPORTS_PATH}/{area}/{model}/{experiment}/{mode}/{log_type}/{file}"

    df = pd.read_csv(fn, comment='#', sep='\t')

    #df = df[(df['layer'] == 0)]
    print(f"Loaded file: {fn}")
    return df, fn


# In[4]:


# computes the total number of injections per layer. df must be a trace df
def get_layer_injections(df, layer=-1):
    if layer >= 0:
        return len(df[df['layer'] == layer])
    else:
        return dict(df.groupby('layer').size())

# returns a list of injected layers
def get_injected_layers(df):
    layers = sorted(df["layer"].unique().tolist())
    return layers

# returns a list of injected layers
def get_injected_inputs(df):
    layers = sorted(df["input_id"].unique().tolist())
    return layers


# In[5]:


# Filters the TP, FN and FP items from the lists
def filter_true_positives(list_oracle, list_work):
    set_work = set(list_work)
    return [o for o in list_oracle if o in set_work]

def filter_false_negatives(list_oracle, list_work):
    set_work = set(list_work)
    return [o for o in list_oracle if not o in set_work]

def filter_false_positive(list_oracle, list_work):
    set_oracle = set(list_oracle)
    """
    for w in list_work:
        if w not in set_oracle:
            print(w)
    """        
    return [w for w in list_work if w not in set_oracle]


# In[6]:


# Computes the recall extracted from the criticality lists. 
# crit_seq and crit_par are dictionaries in the form crit_seq[<fault tag>] = fault criticality
def compute_recall_from_crit_list(crit_seq, crit_par, th=0):
    #if len(crit_seq) == 0:
        #return 0

    critical_list_seq = [k for k in crit_seq if crit_seq[k] > th]

    #critical_list_seq = list(crit_seq.keys()) # if th=0
    critical_list_par = list(crit_par.keys())

    flist_true_positives  = filter_true_positives (critical_list_seq, critical_list_par)
    flist_false_negatives = filter_false_negatives(critical_list_seq, critical_list_par)
    flist_false_positives = filter_false_positive (critical_list_seq, critical_list_par)

    TP = len(flist_true_positives) # true positive size
    FN = len(flist_false_negatives) # false negative size
    
    # [sanity check only]: check for faulse positive faults. cannot happen
    FP = len(flist_false_positives)  # false positive size. we cannot have false positive faults, therefore the precision is always 100% (as it only looks to the "retrieved"/positive data)

    # important
    # when exploring the different criticality thresholds (in compute_recall_from_crit_list), this test is not valid, becase the sequential critical list
    # the crit_seq.keys() will have faults with a criticality above a certain threshold, so when comparing this with the parallel list it will appear as false positive...
    if FP != 0 and th == 0: # check this only if th = 0 
        print(f"False positives detected (FP={FP})")
        #raise ValueError(f"False positives detected (FP={FP})")
   
    precision = TP/(TP + FP) if (TP + FP) != 0 else 0 # will be always 100% as there are no false positives by construction
    recall = TP/(TP + FN)   if (TP + FN) !=0 else 0

    return recall


# In[7]:


# computes the fault criticality. df must be a trace dataframe
# returns a dict in the form dict[<fault tag>] = fault criticality
def compute_fault_criticality(df, layer=-1):
    df_cp = df.copy()

    #df_cp = df_cp[df_cp['layer'] != 6]

    if layer >= 0:
        df_cp = df[df['layer'] == layer]

    grouped = df_cp.groupby('fault_tag').agg(
        injections=('fault_tag', 'size'),
        sdc1_sum=('sdc1', 'sum'),
        #sdc5_sum=('sdc5', 'sum')
    )

    grouped['sdc1_ratio'] = grouped['sdc1_sum'] / grouped['injections']
    #grouped['sdc5_ratio'] = grouped['sdc5_sum'] / grouped['injections']

    # the total number of critical faults 
    sdc1_total_tag = grouped['sdc1_sum'].to_dict()
    #sdc5_total_tag = grouped['sdc1_sum'].to_dict()

    # computes the sdc avf only for the faults that are critical
    sdc1_avf_tag = grouped['sdc1_ratio'][grouped['sdc1_sum'] != 0].to_dict()
    #sdc5_avf_tag = grouped['sdc5_ratio'][grouped['sdc5_sum'] != 0].to_dict()

    injections_tag = grouped['injections'].to_dict()

    return injections_tag, sdc1_total_tag, sdc1_avf_tag


# In[8]:


# returns the total injection time per layer. df must be a batch dataframe
def get_injection_time(df, layer=-1):
    df_cp = df.copy()
    #df_cp = df_cp[df_cp['layer'] != 6]
   
    if layer >= 0:
        df_cp = df[df['layer'] == layer]
    
    # if this column is available, df is a batch df from the parallel approach
    if 'reached_leaves' in df_cp.columns:
        grouped = df_cp.groupby('layer').agg(
            # the total time across all rows (all batches, all layers)
            # each row of col 'injection_time' is the time to sweep the whole tree
            time_total=('injection_time', 'sum'),

            # the mean time per row (i.e., the mean time to sweep the tree for each layer/batch)
            time_mean=('injection_time',  'mean'),

            # the total number of visited nodes
            visited_nodes=('visited_nodes', 'sum'),
            reached_leaves=('reached_leaves', 'sum')
        )
    # or else, df is a batch file from the sequential approach (no information about nodes)
    else:
        grouped = df_cp.groupby('layer').agg(
            time_total=('injection_time', 'sum'),
            time_mean=('injection_time',  'mean'),
        )
        # here we pretend that the sequential approach visited all leaf nodes (one leaf per fault), for all batches
        BATCHES, INJECTIONS = 20, 500
        grouped = grouped.assign(visited_nodes=INJECTIONS*BATCHES) # here we assign 'fake' visited nodes for the sequential batch dataframe
        grouped = grouped.assign(reached_leaves=INJECTIONS*BATCHES)
        
    return grouped['time_total'].to_dict(), grouped['visited_nodes'].to_dict()


# In[9]:


# computes the mean fault criticality from the criticality dict (considers only the faults starting from a threshold)
def compute_mean_avf(crit, crit_th=0):
    values = [v for v in crit.values() if v >= crit_th]
    return sum(values) / len(values) if values else 0


# In[10]:


def compute_div_metric(crit_seq, crit_par):
    seq, us  = list(crit_seq.values()), list(crit_par.values())
    #d, p_value_us = ks_2samp(seq, us)
    d = wasserstein_distance(seq, us)
    return d


# In[11]:


#
# The model name to inspect
#
MODEL="ResNet18"
#MODEL="ResNet50"
#MODEL="ResNeXt101_32X8D"
#MODEL="ResNeXt101_64X4D"
#MODEL="MobileNet_V2"
#MODEL="MobileNet_V3_Large"
#MODEL="GoogLeNet"
#MODEL="Inception_V3"
#MODEL="ShuffleNet_V2_X0_5"
#MODEL="ShuffleNet_V2_X2_0"
#MODEL="deit_tiny"
#MODEL="deit_small"
#MODEL="deit_base"

FOLDER_SEQ="sequential"
FOLDER_PAR="parallel"

# report file area inside the 'report' folder
#AREA="hw-inj-osdim8"  # Faults in Gemmini
#AREA="sw-inj"         # Faults in SW
#AREA="workshop"
AREA="."

# In[14]:


# The experiment folder
#EXPERIMENT_SEQ="exp-rtl-base"
#EXPERIMENT_PAR="exp-rtl-base-v5"


#
#
#
EXPERIMENT_SEQ="exp-rtl-all-toy-inputs-v1"
EXPERIMENT_PAR="exp-rtl-all-toy-inputs-v1"

#EXPERIMENT_SEQ="ws-camp-sw"
#EXPERIMENT_PAR="ws-camp-sw"

EXPERIMENT_SEQ="ws-camp-rtl"
EXPERIMENT_PAR="ws-camp-rtl"

#EXPERIMENT_SEQ="ws-camp-rtl-accumulators"
#EXPERIMENT_PAR="ws-camp-rtl-accumulators"

#EXPERIMENT_SEQ="ws-camp-rtl-ctrl"
#EXPERIMENT_PAR="ws-camp-rtl-ctrl"

#EXPERIMENT_SEQ="ws-camp-gl"
#EXPERIMENT_PAR="ws-camp-gl"

IS_SW_LOG=False

# In[15]:


# [Sequential] loads the trace trace and batch reports
df_trace_seq, fn_trace_seq = load_df(model=MODEL, experiment=EXPERIMENT_SEQ, mode=FOLDER_SEQ, area=AREA, log_type="trace")
df_batch_seq, fn_batch_seq = load_df(model=MODEL, experiment=EXPERIMENT_SEQ, mode=FOLDER_SEQ, area=AREA, log_type="batch")


# In[16]:


# [Parallel] loads the trace trace and batch reports
df_trace_par, fn_trace_par = load_df(model=MODEL, experiment=EXPERIMENT_PAR, mode=FOLDER_PAR, area=AREA, log_type="trace")
df_batch_par, fn_batch_par = load_df(model=MODEL, experiment=EXPERIMENT_PAR, mode=FOLDER_PAR, area=AREA, log_type="batch")


# In[17]:


# filter for a specific layer. use -1 to eval across all layers
EVAL_LAYER=-1


# In[18]:


# gets, for each fault tag: the number of injections, the number of critical faults, the fault criticality dictionary
inj_tag_seq, sdc1_tag_total_seq, sdc1_tag_avf_seq = compute_fault_criticality(df_trace_seq, layer=EVAL_LAYER)

# gets, for each fault tag: the number of injections, the number of critical faults, the fault criticality dictionary
inj_tag_par, sdc1_tag_total_par, sdc1_tag_avf_par = compute_fault_criticality(df_trace_par, layer=EVAL_LAYER)

# computes the fault recall
recall = compute_recall_from_crit_list(sdc1_tag_avf_seq, sdc1_tag_avf_par) # th=0.5/100.0)


# In[19]:


# sanity checking only

# check if the the injected layers are the same for seq and parallel
layers_seq = get_injected_layers(df_batch_seq)
layers_par = get_injected_layers(df_batch_par)
inj_layers_seq = len(layers_seq)
inj_layers_par = len(layers_par)

"""
layer_inj_seq = get_layer_injections(df_trace_seq)
layer_inj_par = get_layer_injections(df_trace_par)
print(layer_inj_seq)
print(layer_inj_par)
"""

if set(layers_seq) != set(layers_par): raise ValueError(f"[Error]: Layer mismatch — layers_seq={set(layers_seq)} vs layers_par={set(layers_par)}")

# check if the the injected inpuyts are the same for seq and parallel
inputs_seq = get_injected_inputs(df_trace_seq)
inputs_par = get_injected_inputs(df_trace_par)

if set(inputs_seq) != set(inputs_par): raise ValueError(f"Input mismatch — seq={set(inputs_seq)} vs par={set(inputs_par)}")


# In[20]:


# the total number of fault injections
inj_total_seq = sum(inj_tag_seq.values())
inj_total_par = sum(inj_tag_par.values())

if inj_total_seq != inj_total_par: raise ValueError(f"The number of injections do not match: seq/par = {inj_total_seq}/{inj_total_par}")


# In[21]:


# the total number of UNIQUE critical faults (the sdc1_avf_seq and sdc1_avf_par only contain crical faults)
sdc1_total_unique_seq = len(sdc1_tag_avf_seq)
sdc1_total_unique_par = len(sdc1_tag_avf_par)


# In[22]:


# the total count of critical faults, including repetitions of the same fault
sdc1_total_seq = sum(sdc1_tag_total_seq.values())
sdc1_total_par = sum(sdc1_tag_total_par.values())
#print(sdc1_total_unique_seq) 
#print(sdc1_total_unique_par)


# In[23]:


# the mean avf computed from the criticality list
avf_mean_seq = compute_mean_avf(sdc1_tag_avf_seq)
avf_mean_par = compute_mean_avf(sdc1_tag_avf_par)


# In[24]:


wasser = compute_div_metric(sdc1_tag_avf_seq, sdc1_tag_avf_par)


# In[25]:


# injection time, visited nodes per layer
inj_time_layer_seq, vn_layer_seq = get_injection_time(df_batch_seq, layer=EVAL_LAYER)
inj_time_layer_par, vn_layer_par = get_injection_time(df_batch_par, layer=EVAL_LAYER)

inj_time_total_seq = sum(inj_time_layer_seq.values())
inj_time_total_par = sum(inj_time_layer_par.values())

speedup = inj_time_total_seq/inj_time_total_par


# In[26]:


# the total number of visited nodes
vn_total_seq = sum(vn_layer_seq.values())
vn_total_par = sum(vn_layer_par.values())

# the time per node
mean_time_per_node_total_seq = inj_time_total_seq/vn_total_seq
mean_time_per_node_total_par = inj_time_total_par/vn_total_par
visited_nodes_ratio = vn_total_seq/vn_total_par


# In[27]:


tab_injections = [["Sequential", inj_total_seq],
                  ["Parallel", inj_total_par]
                 ]

tab_unique_cf = [["Sequential", sdc1_total_unique_seq], 
                 ["Parallel", sdc1_total_unique_par], 
                 ["Missing", sdc1_total_unique_seq - sdc1_total_unique_par],
                 ["Missing", f"{100*(sdc1_total_unique_seq - sdc1_total_unique_par)/sdc1_total_unique_seq:.2f}%"]
                ]

tab_total_cf = [["Sequential", sdc1_total_seq],
                ["Parallel", sdc1_total_par],
                ["Missing", sdc1_total_seq - sdc1_total_par],
                ["Missing", f"{100*(sdc1_total_seq - sdc1_total_par)/sdc1_total_seq:.2f}%"]
               ]

tab_avf = [["Sequential", f"{100*avf_mean_seq:.3f}%"],
           ["Parallel", f"{100*avf_mean_par:.3f}%"],
           ["AVF Error", f"{100*(avf_mean_seq-avf_mean_par)/avf_mean_seq:.2f}%"]
          ]

tab_total_it = [["Sequential", f"{inj_time_total_seq:.2f}"],
                ["Parallel", f"{inj_time_total_par:.2f}"],
                ["Speedup", f"{speedup:.2f}x"]
               ]

tab_visited_nodes = [["Sequential", vn_total_seq],
                     ["Parallel", vn_total_par],
                     ["Reduction", f"{vn_total_seq/vn_total_par:.2f}x"],
                    ]

tab_avg_time_node = [["Sequential", f"{1000*mean_time_per_node_total_seq:.3f}ms"], 
                     ["Parallel", f"{1000*mean_time_per_node_total_par:.3f}ms"],
                     ["Reduction", f"{mean_time_per_node_total_seq/mean_time_per_node_total_par:.2f}x"],
                    ]

tab_inj_layers = [["Sequential", layers_seq], 
                  ["Parallel", layers_par],
                 ]

# In[28]:


print(f"{MODEL} reports:"); print()
print("Files:");print(fn_trace_seq);print(fn_trace_par);print()

if EVAL_LAYER != -1:
    print(f"Layer: {EVAL_LAYER}")
    
print(f"Speedup: {speedup:.2f}x")
print(f"Recall:  {100*recall:.2f}%")
print(f"Wasser:  {wasser}")
print()

# print tables
print(tabulate(tab_injections, headers=["Mode", "Injections"])); print()
print(tabulate(tab_unique_cf,  headers=["Mode", "Unique critical faults"])); print()
#print(tabulate(tab_total_cf,   headers=["Mode", "Critical faults (total)"])); print()
print(tabulate(tab_total_it,   headers=["Mode", "Injection time"])); print()

print(tabulate(tab_avf, headers=["Mode", "AVF"])); print()
print(tabulate(tab_inj_layers, headers=["Mode", "Injected layers"])); print()
