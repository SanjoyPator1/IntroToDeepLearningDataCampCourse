import numpy as np

input_data = np.array([2, 3])

weights = { 'node_0' : np.array([1,1]) , 'node_1' : np.array([-1,1]) , 'output' : np.array([2,-1]) }

node_0_value = (input_data * weights['node_0']).sum()
node_1_value = (input_data * weights['node_1']).sum()

hidden_layer_value = np.array([node_0_value , node_1_value])
print(hidden_layer_value)
#- [5,1] will be printed

output = (hidden_layer_value * weights['output']).sum()
print(output)
#output will be 9
