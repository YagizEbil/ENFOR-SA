import json
from collections import Counter

with open("netlist.json") as f:
    data = json.load(f)

counts = Counter()

"""
369   $and
8   $not
152   $or
244   $xor
"""

for module in data["modules"].values():
    for cell in module.get("cells", {}).values():
        
        if cell["type"] == "$and":  # filter AND gates
            # count all input ports
            num_inputs = sum(len(cell["connections"][port]) 
                             for port in cell["connections"] 
                             if port != "Y")  # exclude output
            counts[f"AND{num_inputs}"] += 1
            

        elif cell["type"] == "$not":  # filter NOT gates
            num_inputs = sum(len(cell["connections"][port]) 
                             for port in cell["connections"] 
                             if port != "Y")  # exclude output
            counts[f"NOT{num_inputs}"] += 1

    
        elif cell["type"] == "$or": 
            num_inputs = sum(len(cell["connections"][port]) 
                             for port in cell["connections"] 
                             if port != "Y")  # exclude output
            counts[f"OR{num_inputs}"] += 1
            
        elif cell["type"] == "$xor":
            num_inputs = sum(len(cell["connections"][port]) 
                             for port in cell["connections"] 
                             if port != "Y")  # exclude output
            counts[f"XOR{num_inputs}"] += 1

print(counts)
