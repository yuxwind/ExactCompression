import os
import sys
import numpy as np

#types = ["100-100-100-100-100"] # "25-25-25" #["800-800"] # [ "25-25", "50-50", "100-100", "200-200", "400-400", "25-25-25", "50-50-50", "100-100-100" ] 
#types = ["100-100-100-100-100"] 
types = ["100-100", "200-200", "400-400", "800-800", "100-100-100", "100-100-100-100", "100-100-100-100-100"]
type_arch = {"25-25": "fcnn2", 
                "50-50": "fcnn2a", 
                "100-100": "fcnn2b", 
                "200-200": "fcnn2c", 
                "400-400": "fcnn2d", 
                "800-800": "fcnn2e", 
                "25-25-25": "fcnn3", 
                "50-50-50": "fcnn3a", 
                "100-100-100": "fcnn3b", 
                "25-25-25-25": "fcnn4", 
                "50-50-50-50": "fcnn4a", 
                "100-100-100-100": "fcnn4b", 
                "100-100-100-100-100": "fcnn5b"}
c0 = np.arange(0,0.00021, 0.000025)
c1 = np.arange(0,0.00041, 0.000025)
l1_reg = { "25-25": [ 0.001 ],
            "50-50": [ 0.0, 0.00015, 0.0003 ],
            "100-100": c1,
            "200-200": c0,
            "400-400": c0,
            "800-800": c0,
            "25-25-25": [0.0003],
            "50-50-50": [0.0003],
            "100-100-100": c0,
            "25-25-25-25": [0.0007],
            "50-50-50-50": [0.0, 0.0002, 0.0003],
            "100-100-100-100": c0,
            "100-100-100-100-100": c0}
type_id = 0
if len(sys.argv) >= 2:
    opt = sys.argv[1]
if len(sys.argv) >= 3:
    type_id = int(sys.argv[2])
if len(sys.argv) >= 4:
    l1_reg[types[type_id]] = [float(sys.argv[3])]

first_network = 0 
nb_networks = 5

if len(sys.argv) >= 5:
    first_network = int(sys.argv[4])
if len(sys.argv) >= 6:
    nb_networks = int(sys.argv[5])
#dataset = "MNIST" # Can also be "CIFAR10-rgb", "CIFAR10-gray", "CIFAR100-rgb"
dataset = "CIFAR10-rgb" # Can also be "CIFAR10-rgb", "CIFAR10-gray", "CIFAR100-rgb"

if opt == 'train':
    train_networks = True
else:
    train_networks = False
if opt == 'new_prune':
    test_new_compression = True
else:
    test_new_compression = False
if opt == 'old_prune':
    test_old_compression = True
else:
    test_old_compression = False

time_limit = 10800 

model_dir = os.path.join('model_dir', dataset)

import pdb;pdb.set_trace()
for idx, type in enumerate(types):
    if idx != type_id:
        continue
    for l1 in l1_reg[type]:
        for network in range(first_network,first_network+nb_networks):
            folder = os.path.join(model_dir, "dnn_"+dataset+"_"+type+"_"+str(l1)+"_"+str(network).zfill(4))

            if train_networks:
                print("python train_fcnn.py --arch " + type_arch[type] + " --save-dir " + folder + " --l1 " + str(l1) + " --dataset " + dataset + " --eval-stable ")
                #os.system("python train_fcnn.py --arch " + type_arch[type] + " --save-dir " + folder + " --l1 " + str(l1) + " --dataset " + dataset + " --eval-stable ")

            if test_old_compression:
                #os.system("python get_activation_patterns.py -b --input " + folder + "/weights.dat" + " --formulation neuron --time_limit " + str(time_limit) + " --dataset " + dataset)
                print("python get_activation_patterns.py -b --input " + folder + "/weights.dat" + " --formulation neuron --time_limit " + str(time_limit) + " --dataset " + dataset)
            if  test_new_compression:
                #os.system("python get_activation_patterns.py -b --input " + folder + "/weights.dat" + " --formulation network --time_limit " + str(time_limit) + " --dataset " + dataset + ' --preprocess_all_samples')
                print("python get_activation_patterns.py -b --input " + folder + "/weights.dat" + " --formulation network --time_limit " + str(time_limit) + " --dataset " + dataset + ' --preprocess_all_samples')
