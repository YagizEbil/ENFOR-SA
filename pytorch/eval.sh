
python scripts/data_proc/eval.py reports/ResNet18/ws-camp-rtl/sequential/trace/sim-xyz-s0-OSDIM8.csv
python scripts/data_proc/eval.py reports/ResNet18/ws-camp-sw/sequential/trace/sim-xyz-s0-SW.csv

# criticality PDF, sequential vs parallel, for all toy input cases (takes around 15min for the sequential)

# RTL ctiticality (Sequential)
python scripts/data_proc/eval_criticality.py reports/ResNet18/exp-rtl-all-toy-inputs-v1/sequential/trace/sim-xyz-s0-OSDIM8.csv

# RTL criticality (Sequential vs Parallel)
python scripts/data_proc/eval_criticality.py reports/ResNet18/exp-rtl-all-toy-inputs-v1/sequential/trace/sim-xyz-s0-OSDIM8.csv reports/ResNet18/exp-rtl-all-toy-inputs-v1/parallel/trace/sim-xyz-s0-OSDIM8.csv