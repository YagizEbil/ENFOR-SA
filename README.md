## Description
This is the repository for the [ENFOR-SA](https://hal.science/hal-05487559v1) fault injection framework. The goal of this tool is to provide accurate fault injection support for systolic array architectures. To do so, we provide a transparent way to map PyTorch-level matmul kernels down to low-level models of systolic arrays. So far, this has been used to simulate transient faults in Convolutional and Attention layers of quantized DNN models for image classification on [ImageNet](https://www.image-net.org/index.php). ENFOR-SA uses the Mesh unit from the [Gemmini](https://github.com/ucb-bar/gemmini) systolic array (see the details in the paper). Faults can be injected in PE registers of RTL models of the array using verilator. 

## Directions

### Install the requirements
The required modules are listed in `requirements.txt`. It's advisable to install them in a virtual environment:

```
python3 -m venv <your env path>
source <your env path>/bin/activate
python3 -m pip install -r requeriments.txt
```

For fault injection tests, you will also need a copy of the validation set of [ImageNet](https://www.image-net.org/index.php).

### Install verilator (only if needed)
Verilator is only necessary if you need to:
- Integrate new Gemmini configs to the framework;
- Recompile the verilated systolic array code for your machine.
Some array configs are already provided in [rtl/designs](rtl/designs). These configs are already compiled in [rtl/verilated](rtl/verilated). 

To use the configurations already provided, you do not need to install verilator. In case you need to [install verilator](https://verilator.org/guide/latest/install.html), make sure to use **version 4.224** to match the HDFIT version (more on this latter). 

## Run tests
Before running experiments, check that the tool passes the basic matmul tests. All experiments and tests must be run from the `pytorch` folder. 

### Test the systolic array
First `cd pytorch`, then: 
```
python tests/test_gemmini.py
```

This tests basic systolic array operations required for matmuls: input preloading, streaming input tensors in the array, and flushing the outputs. All matmul output tensors should match exactly the expected results computed in software.

> [!NOTE]
> The tests take a while to start running when executed for the first time for each new tested array configuration. This is due to the backend compilation of the Gemmini modules and C++ extensions to interface PyTorch with the verilated array model. Latter uses of the same configuration run much faster as PyTorch keeps each compiled configuration cached and no re-compilation is performed (unlsess you change the compilation flags or backend source code).

### Validate against HDIFT
We validate our injection approach by comparing against the [HDFIT](https://intellabs.github.io/HDFIT/) injection procedure. HDFIT relies on verilog code instrumentation for fault injection, and provides ground thruth injection results by instrumenting every assignment directly in verilog code. Run:
```
python tests/test_hdfit.py
```
This should output "Trials not matching HDFIT:  0".

## Run injection experiments
Faults can be injected in RTL models of systolic arrays (directly in PE registers using verilator) or in SW-level (directly in PyTorch tensors). We do not provide any SW-based noise model for errors in SW. Only (naive) random bit flips in the tensors are performed as this aims to provide a baseline for accuracy comparison w.r.t more accurate injection approaches. 

First, make sure you have the validation set of [ImageNet](https://www.image-net.org/index.php) (there's no need to download the train and test sets!). Then follow the next steps.

### Setup the experiment parameters
Experiment setups are defined in the experiment scripts. Examples are available in the [pytorch/experiments](pytorch/experiments) folder, where your experiment configurations must be stored. You can have multiple experiment files stored in this folder. Source a single one in the [`pytorch/experiment.sh`](pytorch/experiment.sh) file.

#### Setup the main simulation script (run.sh)
This is the main execution script ([`run.sh`](pytorch/run.sh)). It loops over each desired model/target layer, loads the selected experiment script and invokes the fault injection tool. 

In this script, set the DNN models to experiment with (the `$models` array) and target layers to inject (`$target_layers`). All other experiment parameters are kept in experiment files, as described next.

#### Setup the experiment parameters (experiment scripts)
Follow the examples of experiment scripts already provided (e.g.,[`pytorch/experiments/experiment_params_rtl.sh`](pytorch/experiments/experiment_params_rtl.sh)). Among other things, the main experiment parameters are as follows:
1. The experiment name: in the `$experiment` variable. This will be the folder name for the report files. The `$sim_alias` variable will be the report file name. Be careful not to mix different experiment types and runs in the same folder/file (results accumulate in the files);
2. The fault model: set the `$fault_model` variable to the desired fault model. Current options are **"rtl"** and **"sw"**;
3. The Gemmini configuration case to be injected (`$gemmini_config_key`). There are many options commented in the scripts;
4. The fault list (`$fault_list`). Examples of lists are provided in [pytorch/fault_lists](pytorch/fault_lists), for the ImageNet dataset and an 8x8 Output Stationary Gemmini config. New fault lists can be generated by calling `python gen_fault_list.py <model name>`. The fault list generation depends on the fault model (and desired array configuration for RTL fault lists). Before generating new fault lists, these options must be set properly in the fault list generation script: just tweak the `USE_SA` and `CONFIG_KEY` variables;
5. The number of faults to inject (`$injections`), the batch size (`$batch_size`), and number of batches (`$nbatches`) in the experiment.

> [!WARNING]
> In the experiment file, the selected fault list must align with the required fault injection abstraction level. For example, for Gemmini injection, set `fault_list=fl_os_dim_8.csv`. For SW injection, set `fault_list=fl_sw.csv`. Selecting the wrong fault list leads to simulation crashes or at least completely wrong results.

### Selecting the fault injection targets
RTL injection can target any registers in the systolic array. This includes:
- Data registers: each PE has two input registers and one output register. For OS arrays, the output register is the accumulator. For WS, the output of each PE is the partial sum passed as input to downstream PEs.
For DNNs, the two input registers hold mapped values of activations and weights;
- Control signals: the `valid` and `propagate` control signals are stored in registers, and are also propagated downstream in systolic fashion. See the paper for more details.

If desired, fault injection can be performed in specific targets by filtering the desired injection points in the full fault list. One can select: 

1. The target layer to offload to the array and inject. For transient faults, a single tile from the layer is offloaded to the array;
2. Target registers (e.g., registers holding weights or activations, outputs or control signals);
3. Specific PE positions (rows and/or columns), and bit positions in the targeted register of the PE.

**Directions:** Target filters can be customized in the file [`pytorch/src/flist/fi_target.py`](pytorch/src/flist/fi_target.py). To inject only a given target type, configure the global dictionary `fault_target` with the desired filters, as shown in the examples provided in the same file.

### Run the experiment
The experiments must be run in the `pytorch` folder. First,`cd pytorch`, then configure the following scripts:

1. Set your ImageNet path:
```
export PATH_IMAGENET=<Path to ImageNet>
```
2. Script [`pytorch/run.sh`](pytorch/run.sh): Select the desired models and layers to inject;
3. Folder [pytorch/experiments](pytorch/experiments): Create and configure an experiment script, according to the examples provided (e.g., [`experiment_params_rtl.sh`](pytorch/experiments/experiment_params_rtl.sh));
4. Script [`pytorch/experiment.sh`](pytorch/experiment.sh): Source your experiment script (e.g., `experiment_params_rtl.sh`);
5. Hit `./run.sh`.

The experiment results are logged in `pytorch/reports`. Two types of report files are generated: the `batch` folder summarizes per-batch data (e.g., total number of critical faults per batch per injected layer). The `trace` folder logs data for each individual injected fault and input.

### Example of trace file for RTL injections
The table below shows an example of a trace file generated for RTL injections. The file stores the full fault position: the injected layer, tile position offloaded to the array, target PE, the type of register (inputs, outputs or control) injected in the PE (column “target”) and target bit. 

The main classification effects of the fault is shown in columns *sdc1* and *sdc5*. 
The sdc1 flag is asserted if the fault caused the Top-1 classification label to diverge from the golden label. The sdc5 flag indicates a more aggressive fault, as it is asserted if the Top-1 label is not any of the Top-5 golden labels. For quantized CNN models, we also keep track wheather the fault was masked by the array right during the HW matmul (*gemm_msk*) or during any SW phase of the requantization steps (scaling - *scale_msk*, rounding - *round_msk* or clamping - *clamp_msk*). If the fault is not masked in any of these stages, it gets exposed to the SW by corrupting the layer's output. This effect in the layer may or may not lead to SDCs.

| input_id | fault_tag | layer | tile_row | tile_col | target | pe_row | pe_col | bit | sdc1 | sdc5 | gemm_msk | scale_msk | round_msk | clamp_msk | 
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 25247 | 42 | 0 | 6 | 1206 | 4 | 0 | 2 | 27 | 0 | 0 | 0 | 0 | 0 | 0 | 1
| 49673 | 7 | 1 | 2 | 44 | 1 | 3 | 7 | 4 | 0 | 0 | 0 | 0 | 0 | 1 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

Check the examples [here](pytorch/reports/README.md) on how to process the trace files.

## Porting new configurations
To port new Gemmini configurations, check the documentation [here](rtl/README.md).

## API for the systolic array handler
Check the [API to connect PyTorch with the verilated Gemmini modules.](pytorch/src/gemmini/README.md)

## API for fault injection loops
Check the [API to load models instrumented for fault injection and run injection loops.](pytorch/README.md)

## Citation
To cite this work:
```
@inproceedings{Tonetto2026ENFORSA,
  author    = {Rafael Billig Tonetto and Marcello Traiola and Fernando Fernandes and Angeliki Kritikakou},
  title     = {ENFOR-SA: End-to-end Cross-layer Transient Fault Injector for Efficient and Accurate DNN Reliability Assessment on Systolic Arrays},
  booktitle = {Proceedings of the IEEE VLSI Test Symposium (VTS)},
  year      = {2026},
  month     = apr,
  address   = {Napa Valley, CA, United States},
  note      = {HAL Id: hal-05487559},
  url       = {https://hal.science/hal-05487559v1},
}
```