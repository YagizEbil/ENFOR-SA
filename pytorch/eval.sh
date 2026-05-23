# reports/workshop/ResNet18/exp-rtl-v1/parallel/trace/sim-xyz-s0-OSDIM8.csv

#python scripts/data_proc/eval_pe.py $1
python scripts/data_proc/eval_criticality.py $1 $2
#python scripts/data_proc/eval.py $1


# python scripts/data_proc/eval_criticality.py reports/workshop/ResNet18/exp-rtl-v1/sequential/trace/sim-xyz-s0-OSDIM8.csv reports/workshop/ResNet18/exp-rtl-v1/parallel/trace/sim-xyz-s0-OSDIM8.csv