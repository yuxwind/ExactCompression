#!/usr/bin/python

"""
    Sample Run -
    ./get_activation_patterns.py -i weight_files/XOR.dat -c
    ./get_activation_patterns.py -i ../forward_pass/activation_pattern/models/fcnn_run_54/weights.dat    | tee ../forward_pass/activation_pattern/models/fcnn_run_54/bounds.log
    ./get_activation_patterns.py -i ../forward_pass/activation_pattern/models/fcnn_run_54/weights.dat -a | tee ../forward_pass/activation_pattern/models/fcnn_run_54/bounds.log
    ./get_activation_patterns.py -i ../forward_pass/activation_pattern/models/fcnn_run_132/weights.dat -m 0.001 | tee ../forward_pass/activation_pattern/models/fcnn_run_132/bounds_width_0.001.log
    ./get_activation_patterns.py -i ../forward_pass/activation_pattern/models/fcnn_run_132/weights.dat -x local_stability/mnist_sample_class_0.dat -L 0.001 | tee ../forward_pass/activation_pattern/models/fcnn_run_132/bounds_xbar_0_width_0.001.log

    This function calls gurobipy on a deep neural network weights and biases to
    frame an optimisation problem and then obtain the solution to decide the
    minimum and maximum of each node at each layer. The input is assumed to be
    bounded. Then these bounds are used to calculate the activation pattern
    at each of the nodes barring the input nodes. It also write two files in the
    same directory as the weights file -
    1) inactive_nodes_file - File which specifies which layer and index, the
        units are off
    2) activation_pattern_file - File which contains all possible activation
        patterns of the network

    Modes in which the functions work:
    1) Normal mode - Solve exactly for till total_layers-2 and get a feasible
        solution at total_layers-1 to see the maxima and minima. If no maxima
        or minima is found at the top_layer-1, that means the solution does
        not exist

    2) Approx mode - Get a feasible solution at all layers and use
        model.objBound which is the solution of LP - the relaxed version of
        the MILP to get the H and Hbars.

    Note:
    We never solve for the final layers for classification models since that
    is the softmax layer and is required for classification
"""

import argparse
import os
import numpy as np
import time
from gurobipy import *
import re

import random

from common.io import mkpath, mkdir

accuracy = None

time_before = time.time()

################################################################################
# Line Parser
################################################################################
def parse_line(line):
    my_list = re.split(', |,| |\[|\]|\];', line)
    # Ignore empty strings
    # my_list = filter(None, my_list)
    # <THIAGO>
    new_list = []
    for i in my_list:
        if len(i)>0:
            new_list.append(i)
    my_list = new_list
    # </THIAGO>
    output  = np.array(my_list)

    return output


################################################################################
# Returns the actual index in the matrix
################################################################################
def parse_file(input):

    word1 = "levels = "
    word2 = "n = ["
    word3 = "W ="
    word4 = "B ="
    word5 = "];"


    # Output variables
    layers = 0   # layer number at which we have the output
    weights = [] # list whose each elements contains weight matrices
    bias    = [] # list whose each element contain biases

    # IMP NOTE -
    # Bias should be in the form of matrices. eg if we have 2 bias term
    # for a layer, the bias element shape should be (2,1) and not (2, )

    # Assumes that each row of each weight matrix is written in 1 line
    weight_flag = False
    weight_line_cnt  = 0

    # Assumes that bias for 1 layer is written in 1 line
    bias_flag   = True
    bias_layer_cnt = 0

    with open(args.input, 'r') as fp:
        for cnt, line in enumerate(fp):
            # Remove trailing characters
            line = line.strip()

            # if not an empty line
            if(len(line) > 0):

                # Comment found. Skip
                if (line[0:2] == "//"):
                    if (line[0:3] == "//C"):
                        tokens = line.split()
                        global accuracy
                        accuracy = float(tokens[3].split("%")[0])
                    pass

                elif (word1.lower() in line):
                    layers = int(line[len(word1)])

                elif (word2.lower() in line):
                    temp            = line[len(word2):-2]
                    nodes_per_layer = parse_line(temp).astype(int)
                    line_bins       = np.cumsum(nodes_per_layer[1:])

                elif (word3.lower() in line.lower()):
                    weight_flag = True

                elif (word4.lower() in line.lower()):
                    bias_flag = True

                else:
                    if (weight_flag):
                        if (word5 in line):
                            weight_flag = False
                        else:
                            # These will be weights
                            # Get which weight matrix the current line goes
                            index = np.digitize(weight_line_cnt, line_bins) #np digitize is 1 ordered

                            # Need a new weight matrix
                            if (weight_line_cnt == 0 or weight_line_cnt in line_bins):
                                weights.append(np.zeros((nodes_per_layer[index + 1], nodes_per_layer[index])))
                                row_cnt = 0

                            # row_cnt keeps track of the row of the weight matrix to write this line
                            weights[index][row_cnt, :] = parse_line(line).astype(np.float)

                            row_cnt += 1
                            weight_line_cnt += 1

                    elif (bias_flag):
                        if (word5 in line):
                            bias_flag = False
                        else:
                            # These will be biases
                            temp = parse_line(line).astype(np.float)
                            bias.append(np.transpose([temp]))

    return layers, nodes_per_layer, weights, bias


################################################################################
# Print bounds
################################################################################
def print_bounds(tot_layers, nodes_per_layer, bounds):
    max_nodes = np.max(nodes_per_layer[1:])

    np.set_printoptions(threshold=np.inf, formatter={'float': lambda x: "{0:0.2f}".format(x)})
    print("\nMaxima for the nodes")
    print(bounds[1:, 0:max_nodes, 0])
    print("Minima of the nodes")
    print(bounds[1:, 0:max_nodes, 1])

    r,c = np.where(bounds[1:, 0:max_nodes, 0] <= 0)

    print("")
    #print("------------------------------------------------------------------------")
    if (r.shape[0] > 0):
        print("Number_stably_inactive_nodes {}".format(r.shape[0]))
        #print("------------------------------------------------------------------------")
    else:
        print("Number_stably_inactive_nodes {}".format(0))

    with open(os.path.join(os.path.dirname(args.input), inactive_nodes_file), 'w') as the_file:
        for i in range(r.shape[0]):
            l = r[i] + 1   #+1 since we have ignored the input nodes while printing
            u = c[i]
            #print("(%d, %d)" %(l, u))
            the_file.write(str(l) + " " + str(u) + "\n")

    r,c = np.where(bounds[1:, 0:max_nodes, 1] <= 0)

    #print("------------------------------------------------------------------------")
    if (r.shape[0] > 0):
        print("Number_stably_active_nodes   {}".format(r.shape[0]))
        #print("------------------------------------------------------------------------")
    else:
        print("Number_stably_active_nodes   {}".format(0))


################################################################################
# Argument Parsing
################################################################################
ap = argparse.ArgumentParser()
ap.add_argument    ('-i', '--input'                                  , help = 'path of the input dat file'           , default='./weight_files/XOR.dat')
ap.add_argument    ('-a', '--approx'           , action='store_true'  , help = 'use approx algorithm for calculation (default: False)')
ap.add_argument    ('-b', '--bounds_only_flag' , action='store_true'  , help = 'bounds only flag (default: False)')
ap.add_argument    ('--preprocess_all_samples' , action='store_true'  , help = 'preprocess the neuron stability using all training samples(default: False)')
ap.add_argument    ('--preprocess_partial_samples' , action='store_true'  , help = 'preprocess the neuron stability using partial training samples(default: False)')
ap.add_argument    ('-c', '--classify_flag'    , action='store_false' , help = 'classification flag (default: True)')
ap.add_argument    ('-m', '--maximum'          , type=float           , help = 'maxima of the nodes (default: 1)'     , default='1')
ap.add_argument    ('-x', '--xbar_file'                               , help = 'Center of the individual input nodes in csv format. Could be an image of the validation set.')
ap.add_argument    ('-L', '--width_around_xbar', type=float           , help = 'Width around xbar of each individual node. (default: 0.0001)' ,default='0.0001')
ap.add_argument    ('-f', '--formulation'                             , help = 'Formulation to be used: neuron, layer, network (default: network)', default='network')
ap.add_argument    ('-F', '--feasible'                                , help = 'Injection of feasible solution based on network input: relaxation, random, off (default: relaxation; not available for neuron formulation)', default='relaxation')
ap.add_argument    ('-t', '--time_limit', type=float                  , help = 'Time limit in seconds to conclude MILP solve (default: None)', default=None)
ap.add_argument(        '--dataset'        , dest='dataset', type=str         , default='MNIST'     , help='Dataset to be used (default: MNIST)')
args = ap.parse_args()


determine_stability_per_network = (args.formulation=='network')
determine_stability_per_layer = (args.formulation=='layer')
determine_stability_per_unit = (args.formulation=='neuron')

if args.formulation=='neuron':
    args.feasible = 'off'

inject_relaxed_solution = (args.feasible=='relaxation')
inject_random_solution = (args.feasible=='random')

################################################################################
# Parameters
################################################################################
# Bound on the input nodes
# Input_min should not be negated as the hidden units since these bounds are
# directly used by the constraints of MILP formulation
input_min = 0
input_max = args.maximum

# If approx method fails, solution value
approx_method_no_solution_val = 1

# Optimisation Display
disp_opt = False

# Saving options
save_model  = False
save_folder = "lp_models"

# Activation Pattern options
show_activations = False
print_freq       = 1000


################################################################################
# Initialisations
################################################################################
layers, nodes_per_layer, weights, bias = parse_file(args.input)

# Total number of layers including input is layers+1
tot_layers = layers+1
max_nodes  = np.max(nodes_per_layer)

# bounds contain two values for every node
# 0 index in bounds is for maxima and 1 index in bounds is for minima
bounds =  12.34*np.ones((tot_layers, max_nodes, 2))

# Initialize bounds for layer 0 (input layer)
if (args.xbar_file is None):
    bounds[0, 0:nodes_per_layer[0], 0] = input_max
    bounds[0, 0:nodes_per_layer[0], 1] = input_min

    # Inactive_nodes and activation patterns file
    inactive_nodes_file     = "inactive_input_" + str(input_min) + "_" + str(input_max) + ".dat"
    active_nodes_file       = "active_input_"   + str(input_min) + "_" + str(input_max) + ".dat"
    activation_pattern_file = "activation_pattern_abhinav_input_" + str(input_min) + "_" + str(input_max) + ".dat"

else:
    input_nodes_center = np.genfromtxt(args.xbar_file,  delimiter=',')
    width_around_xbar  = args.width_around_xbar

    assert input_nodes_center.shape[0] == nodes_per_layer[0]

    print("Ignoring values supplied by command argument maximum !!!")
    print("Putting center around the values given by the file {}".format(args.xbar_file))
    print("Width around xbar = {:.5f}".format(width_around_xbar))

    for i in range(nodes_per_layer[0]):
        bounds[0, i, 0] = input_nodes_center[i] + width_around_xbar
        bounds[0, i, 1] = input_nodes_center[i] - width_around_xbar

    # Inactive_nodes and activation patterns file
    inactive_nodes_file     = "inactive_center_" + os.path.basename(args.xbar_file[:-4]) + "_width_around_xbar_" + str(width_around_xbar) + ".dat"
    active_nodes_file       = "active_center_"   + os.path.basename(args.xbar_file[:-4]) + "_width_around_xbar_" + str(width_around_xbar) + ".dat"
    activation_pattern_file = "activation_pattern_abhinav_center_" + os.path.basename(args.xbar_file[:-4]) + "_width_around_xbar_" + str(width_around_xbar) + ".dat"


# Intialisation list for variables
lst = []
# ***************************** Changed here *******************************
for i in range(tot_layers):
    for j in range(nodes_per_layer[i]):
        lst += [(i, j)]


print("\n\n===============================================================================");
print("Input File    = %s" %(args.input))
print("Num of layers = %d" %(layers))
print("Nodes per layer")
print(nodes_per_layer)
print("===============================================================================");
#print("\nWeights")
#print(weights)
#print("\nBias")
#print(bias)

print("")
if (args.approx):
    print("\nSolving using the approx method ...\n\n")


print("Lower bound of input node = {}".format(input_min))
print("Upper bound of input node = {}".format(input_max))

# ***************************** Changed here *******************************
if (args.classify_flag):
    print("Classification Model. Ignore the last layer since that is used for classes")
    run_till_layer_index =  tot_layers-1
else:
    print("Not a Classification Model. Running for all layers including the last layer")
    run_till_layer_index =  tot_layers
print("------------------------------------------------------------------------")


def mycallback(model, where):
    global positive_solution

    if where == GRB.Callback.MIP:
        objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
        if objbnd < 0:
            #print(" * NEGATIVE BOUND *")
            model.terminate()
            pass
        if objbnd < GRB.INFINITY and positive_solution:
            #print(" * POSITIVE BOUND WITH POSITIVE SOLUTION *", objbnd)
            model.terminate()
    elif where == GRB.Callback.MIPSOL:
        nodecnt = model.cbGet(GRB.Callback.MIPSOL_NODCNT)
        obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        if obj > 0:
            #print(" * (POSITIVE SOLUTION) *")
            positive_solution = True

def layercallback(model, where):
    global p, q, i, nodes_per_layer, positive_units, negative_units
    global h, g

    if where == GRB.Callback.MIPSOL:
        print("FOUND A SOLUTION")
        p_value = model.cbGetSolution(p)
        q_value = model.cbGetSolution(q)
        g_value = model.cbGetSolution(g)
        for n in range(nodes_per_layer[i]):
            if p_value[n] == 1:
                positive_units.add(n)
                model.cbLazy(p[n] == 0)
                #print("+",n,g_value[i,n])
            elif q_value[n] == 1:
                negative_units.add(n)
                model.cbLazy(q[n] == 0)
                #print("-",n,g_value[i,n])
            else:
                pass
                #print("?",n,g_value[i,n])
    elif where == GRB.Callback.MIP:
        objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
        print("BOUND:", objbnd)
        if objbnd<0.5:
            model.terminate()
    elif where == GRB.Callback.MIPNODE:
        print("MIPNODE")
        vars = []
        values = []

        if inject_relaxed_solution:
            for input in range(nodes_per_layer[0]):
                vars.append(h[0,input])
            values = model.cbGetNodeRel(vars)
            model.cbSetSolution(vars,values)
        elif inject_random_solution:
            for input in range(nodes_per_layer[0]):
                vars.append(h[0,input])
                values.append(bounds[0, input, 0] + random.random()*(bounds[0, input, 1]-bounds[0, input, 0]))
            model.cbSetSolution(vars,values)

        #obj = model.cbUseSolution()
        #print("GOT",obj)

def networkcallback(model, where):
    global p, q, i, nodes_per_layer, positive_units, negative_units
    global h
    global lst
    
    if where == GRB.Callback.MIPSOL:
        print("FOUND A SOLUTION")
        p_value = model.cbGetSolution(p)
        q_value = model.cbGetSolution(q)
        for (m,n) in p_lst:
            if p_value[m,n] == 1:
                positive_units.add((m,n))
                model.cbLazy(p[m,n] == 0)
                #print("+",m,n)
        for (m,n) in q_lst:
            if q_value[m,n] == 1:
                negative_units.add((m,n))
                model.cbLazy(q[m,n] == 0)
                #print("-",m,n)
    elif where == GRB.Callback.MIP:
        objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
        print("BOUND:", objbnd)
        if objbnd<0.5:
            model.terminate()
    elif where == GRB.Callback.MIPNODE:
        print("MIPNODE")
        vars = []
        values = []

        if inject_relaxed_solution:
            for input in range(nodes_per_layer[0]):
                vars.append(h[0,input])
            values = model.cbGetNodeRel(vars)
            model.cbSetSolution(vars,values)
        elif inject_random_solution:
            for input in range(nodes_per_layer[0]):
                vars.append(h[0,input])
                values.append(bounds[0, input, 0] + random.random()*(bounds[0, input, 1]-bounds[0, input, 0]))
            model.cbSetSolution(vars,values)
           
        #obj = model.cbUseSolution()
        #print("GOT",obj)

################################################################################
# Finding the bounds
################################################################################
network = args.input[:args.input.rfind("/")] #args.input.split("/")[0]
print("Network",network)
print("Accuracy",accuracy)
#f = open("RESULTS.txt","a+")

NOPRE       = 'results-no_preprocess'
ALLPRE      = 'results-preprocess_all'
PARTPRE     = 'results-preprocess_partial'
OLD         = 'results-old-approach'
rst_dir     = './results/'
cnt_rst     = 'counting_results/'
stb_neuron  = 'stable_neurons/'

if args.formulation == 'neuron':
    tag = OLD
else:
    if args.preprocess_all_samples:
        tag = ALLPRE
    elif args.preprocess_partial_samples:
        tag = PARTPRE
    else:
        tag = NOPRE

rst_dir = mkdir(os.path.join(rst_dir, args.dataset, tag, cnt_rst))
exp_name = os.path.basename(os.path.dirname(args.input))
stable_neurons_path = mkpath(os.path.join(rst_dir, args.dataset, tag, stb_neuron, exp_name + '.npy'))
f = open(mkpath(os.path.join(rst_dir, exp_name + '.txt')), "a+")
f.write(network+", "+str(accuracy)+", , ")

timeouts = 0
stably_active = {}
stably_inactive = {}

timed_out = False

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random

p_lst = [(i,j) for i in range(1,tot_layers) for j in range(nodes_per_layer[i]) ]
q_lst = [(i,j) for i in range(1,tot_layers) for j in range(nodes_per_layer[i]) ]
print(-1,len(p_lst),len(q_lst))
last_size = len(p_lst)+len(q_lst)
last_update = -1
max_nonupdates = 1

remove_p = True
remove_q = True

normalize = transforms.Normalize(mean=[0], std=[1]) #Images are already loaded in [0,1]
transform_list = [transforms.ToTensor(), normalize]
if args.dataset == "MNIST": 
    data = datasets.MNIST(root='./data', train=True, transform=transforms.Compose(transform_list), download=True)
elif args.dataset == "CIFAR10-gray" or args.dataset == "CIFAR10-rgb":
    data = datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose(transform_list), download=True)
elif args.dataset == "CIFAR100-rgb":
    data = datasets.CIFAR100(root='./data', train=True, transform=transforms.Compose(transform_list), download=True)
n = data.__len__()
max_nonupdates = 10
to_preprocess_partial = args.preprocess_partial_samples
to_preprocess_all = args.preprocess_all_samples
if to_preprocess_partial:
    if not determine_stability_per_unit:
      for i in range(n):
        #(img, target) = data.__getitem__(random.randint(0,n))
        (img, target) = data.__getitem__(i)
        imgf = torch.flatten(img)
        input = [imgf[j].item() for j in range(nodes_per_layer[0])]
        for l in range(1,tot_layers):
            output = []
            for j in range(nodes_per_layer[l]):
                g = bias[l-1][j][0] + sum([ weights[l-1][j,k]*input[k] for k in range(nodes_per_layer[l-1]) ])
                if g>0 and (l,j) in p_lst and remove_p:
                    p_lst.remove((l,j))
                elif g<0 and (l,j) in q_lst and remove_q:
                    q_lst.remove((l,j))
                output.append(max(0,g))
            input = output
        print(i, len(p_lst), len(q_lst))
        size = len(p_lst)+len(q_lst)
        if size < last_size:
            last_size = size
            last_update = i
        if len(p_lst)+len(q_lst) < 1 or i > last_update + max_nonupdates:
            #print(p_lst, q_lst)
            print(i, last_update, max_nonupdates)
            break
if to_preprocess_all:
    stable_from_sample_path = os.path.join(os.path.dirname(args.input), 'stable_neurons.npy')
    stable_from_sample = np.load(stable_from_sample_path, allow_pickle=True).item()
    q_lst_ = stable_from_sample['stably_active'].squeeze()
    p_lst_ = stable_from_sample['stably_inactive'].squeeze()
    if len(q_lst_.shape) == 1:
        q_lst_ = q_lst_[None,:]
    if len(p_lst_.shape) == 1:
        p_lst_ = p_lst_[None,:]
    q_lst = [(q_lst_[i,0], q_lst_[i,1]) for i in range(q_lst_.shape[0]) ]
    p_lst = [(p_lst_[i,0], p_lst_[i,1]) for i in range(p_lst_.shape[0]) ]
remaining = len(p_lst)+len(q_lst)

for i in range(1,run_till_layer_index):

    stably_active[i] = []
    stably_inactive[i] = []

    # for each node in the layer
    for j in range(nodes_per_layer[i]):

        if determine_stability_per_unit and not timed_out: 

            #print("*** Layer %d Node %d ***" %(i,j))

            # Create a new model
            model = Model("mip1")

            if (not(disp_opt)):
                # Donot display output solving
                # https://stackoverflow.com/a/37137612
                model.params.outputflag = 0

            # Create variables
            g    = model.addVars(lst, lb=-GRB.INFINITY, name="g")
            h    = model.addVars(lst, lb=0.0          , name="h")
            hbar = model.addVars(lst, lb=0.0          , name="hbar")
            z    = model.addVars(lst, vtype=GRB.BINARY, name="z")

            # Add constraints for the current layer:
            name = "c_" + str(i) + str(j)
            model.addConstr(quicksum(weights[i-1][j,k] * h[i-1,k] for k in range(nodes_per_layer[i-1])) + bias[i-1][j] - g[i,j] == 0, name + "_1")

            # Specify bounds of the input variables.
            # 0 index in bounds is for maxima and 1 index in bounds is for minima
            model.addConstrs( h[0, k] <= bounds[0, k, 0] for k in range(nodes_per_layer[0]))
            model.addConstrs( h[0, k] >= bounds[0, k, 1] for k in range(nodes_per_layer[0]))


            # Add constraints for all the nodes starting from the input layer till nodes before this layer
            for m in range(1, i):
                for n in range(nodes_per_layer[m]):
                    name = "c_" + str(m) + str(n)
                    model.addConstr(quicksum(weights[m-1][n,k] * h[m-1,k] for k in range(nodes_per_layer[m-1])) + bias[m-1][n] - g[m,n] == 0, name + "_1")
                    model.addConstr(g[m,n]    == h[m,n] - hbar[m,n], name + "_2")
                    model.addConstr(h[m,n]    <= bounds[m, n, 0] * z[m,n], name + "_3")
                    model.addConstr(hbar[m,n] <= bounds[m, n, 1] * ( 1 - z[m,n] ), name + "_4")

                    if n in stably_active[m]:
                        model.addConstr(z[m,n] == 1)
                    if n in stably_inactive[m]:
                        model.addConstr(z[m,n] == 0)

            # Set objectives for the two optimisation problems - maxima and minima
            # maxima is zero while minima is one
            for k in range(2):
                if timed_out:
                    break

                if(k == 0):
                    pass #cut = model.addConstr(g[i,j] >= 0)
                else:
                    pass #model.remove(cut)
                    #model.addConstr(g[i,j] <= 0)

                # If approx mode where we only need to see one feasible solution the bounds
                # ***************************** Changed here *******************************
                if(args.approx):
                    # https://www.gurobi.com/documentation/8.1/refman/parameters.html
                    model.setParam(GRB.Param.SolutionLimit, 1)
                    model.setParam(GRB.Param.Cutoff, 0.0)
                    #if k==0:
                    #    model.setParam(GRB.Param.Cutoff, abs(1e-4*max_unit))
                    #else:
                    #    model.setParam(GRB.Param.Cutoff, abs(1e-4*min_unit))
                    model.setParam(GRB.Param.TimeLimit, 30)

                
                if(k == 0):
                    model.setObjective(g[i,j] , GRB.MAXIMIZE)
                else:
                    model.setObjective(-g[i,j], GRB.MAXIMIZE)

                try:
                    print("SOLVING FOR",network,i,j,k)
                    positive_solution = False
                    #model.optimize()
                    if args.time_limit != None:
                        time_left = args.time_limit-time.time()+time_before
                        if time_left > 0:
                            model.setParam(GRB.Param.TimeLimit, time_left)
                    if args.time_limit == None or time_left > 0:
                        model.optimize(mycallback)
                    else:
                        timed_out = True
                    if args.time_limit != None and time.time()-time_before > args.time_limit:
                        timed_out = True

                    #print(model.Runtime)

                except GurobiError:
                    print("1 Error reported")

                if not timed_out:

                    if (model.status == GRB.Status.OPTIMAL):
                        bounds[i, j, k] = model.objVal
                    elif (model.status == GRB.Status.SOLUTION_LIMIT):
                        print('Solution Limit Reached')
                        bounds[i, j, k] = model.objBound
                        timeouts = timeouts + 1

                    elif (model.status == GRB.Status.TIME_LIMIT):
                        print('Time Limit Reached')
                        bounds[i, j, k] = model.objBound

                    elif (model.status == GRB.Status.CUTOFF):
                        print('Cutoff Reached')
                        bounds[i, j, k] = model.objBound

                    elif not timed_out:
                        print ("Some other condition found: ", end="")
                        #bounds[i, j, k] = 1 # approx_method_no_solution_val
                        bounds[i, j, k] = model.objBound
                        #print("Bound coming out:", model.objBound)
    
                    if (model.status == GRB.Status.INF_OR_UNBD):
                        print('Model is infeasible or unbounded')
                    elif (model.status == GRB.Status.INFEASIBLE):
                        print('Model is infeasible')
                    elif (model.status == GRB.Status.UNBOUNDED):
                        print('Model is unbounded')
                    else:
                        print('Optimization ended with status %d' % model.status)
                        # Callback termination leads here

                if (disp_opt):
                        print("Objective = %f\n\n" %(model.ObjVal))
                if (save_model):
                    model.write(os.path.join(save_folder, "model_" + str(i) + str(j) + "_" + str(k) +".lp"))

    if determine_stability_per_unit:
        max_g = 0
        max_ng = 0
        for j in range(nodes_per_layer[i]):
            #print(bounds[i,j,0],bounds[i,j,1])
            if bounds[i,j,0] < GRB.INFINITY and bounds[i,j,0] > max_g:
                max_g = bounds[i,j,0]
            if bounds[i,j,1] < GRB.INFINITY and bounds[i,j,1] > max_ng:
                max_ng = bounds[i,j,1]
        #print(max_g, max_ng)

        for j in range(nodes_per_layer[i]):
            for k in range(2):
                    #print(j,k, bounds[i,j,k])
                    if k==0:
                        if bounds[i,j,0] < 0: #<= 1e-5*max_g: # MAX
                          stably_inactive[i].append(j)
                          bounds[i,j,0] = 1
                    else:
                        if bounds[i,j,1] < 0: #<= 1e-5*max_ng: # MIN
                          stably_active[i].append(j)
                          bounds[i,j,1] = 1
                          if j in stably_inactive[i]:
                              print("Numerical error: unit is both stably active and inactive")


    if determine_stability_per_layer and not timed_out:
            # Create a new model
            model = Model("mip1")

            if (not(disp_opt)):
                # Donot display output solving
                # https://stackoverflow.com/a/37137612
                model.params.outputflag = 0

            # Create variables
            g    = model.addVars(lst, lb=-GRB.INFINITY, name="g")
            h    = model.addVars(lst, lb=0.0          , name="h")
            hbar = model.addVars(lst, lb=0.0          , name="hbar")
            z    = model.addVars(lst, vtype=GRB.BINARY, name="z")

            p = model.addVars([k for k in range(nodes_per_layer[i])], name="p")
            q = model.addVars([k for k in range(nodes_per_layer[i])], name="q")

            # Specify bounds of the input variables.
            # 0 index in bounds is for maxima and 1 index in bounds is for minima
            model.addConstrs( h[0, k] <= bounds[0, k, 0] for k in range(nodes_per_layer[0]))
            model.addConstrs( h[0, k] >= bounds[0, k, 1] for k in range(nodes_per_layer[0]))


            for j in range(nodes_per_layer[i]):
              max_unit = bias[i-1][j]
              min_unit = bias[i-1][j]
              for jj in range(nodes_per_layer[i-1]):
                  impact = weights[i-1][j,jj]*bounds[i-1,jj,0]
                  if impact > 0:
                      max_unit = max_unit + impact
                  else:
                      min_unit = min_unit + impact
              bounds[i,j,0] = max(max_unit,1)
              bounds[i,j,1] = max(-min_unit,1)

            # Add constraints for all the nodes starting from the input layer till nodes in this layer
            for m in range(1, i+1):
                for n in range(nodes_per_layer[m]):

                    name = "c_" + str(m) + str(n)
                    model.addConstr(quicksum(weights[m-1][n,k] * h[m-1,k] for k in range(nodes_per_layer[m-1])) + bias[m-1][n] - g[m,n] == 0, name + "_1")
                    model.addConstr(g[m,n]    == h[m,n] - hbar[m,n], name + "_2")
                    model.addConstr(h[m,n]    <= 2*bounds[m, n, 0] * z[m,n], name + "_3")
                    model.addConstr(hbar[m,n] <= 2*bounds[m, n, 1] * ( 1 - z[m,n] ), name + "_4")

            for n in range(nodes_per_layer[i]):
                model.addConstr(p[n] <= z[i,n])
                model.addConstr(q[n] <= 1-z[i,n])

            model.setObjective(quicksum(p[k]+q[k] for k in range(nodes_per_layer[i])) , GRB.MAXIMIZE)
   
            try:
                    print("SOLVING FOR",network,i)
                    positive_units = set()
                    negative_units = set()
                    model.params.LazyConstraints = 1
                    model.params.StartNodeLimit = 1
                    if args.time_limit != None:
                        time_left = args.time_limit-time.time()+time_before
                        if time_left > 0:
                            model.setParam(GRB.Param.TimeLimit, time_left)
                    if args.time_limit == None or time_left > 0:
                        model.optimize(layercallback) 
                    else:
                        timed_out = True
                    if args.time_limit != None and time.time()-time_before > args.time_limit:
                        timed_out = True
            except GurobiError:
                    print("2 Error reported")

            for n in range(nodes_per_layer[i]):
                if n in positive_units and not n in negative_units:
                    stably_active[i].append(n)
                elif n in negative_units and not n in positive_units:
                    stably_inactive[i].append(n)              

    if not determine_stability_per_network:

      if not timed_out:

        print("Layer %d Completed..." %(i))


        matrix_list = []
        for j in stably_active[i]:
            matrix_list.append([weights[i-1][j,k] for k in range(nodes_per_layer[i-1])])
        print("Active: ", stably_active[i])

        rank = np.linalg.matrix_rank(ny.array(matrix_list).astype(float64))

        #print("Active rank: ", rank, "out of", len(stably_active[i]))
        print("Inactive: ", stably_inactive[i])
        f.write(str(len(stably_active[i]))+", "+str(rank)+", "+str(len(stably_inactive[i]))+",, ")
      else:
        f.write("-, -, -,, ")


if determine_stability_per_network:
            for i in range(1,run_till_layer_index):

              stably_active[i] = []
              stably_inactive[i] = []

              for j in range(nodes_per_layer[i]):
                max_unit = bias[i-1][j]
                min_unit = bias[i-1][j]
                for jj in range(nodes_per_layer[i-1]):
                  impact = weights[i-1][j,jj]*bounds[i-1,jj,0]
                  if impact > 0:
                      max_unit = max_unit + impact
                  else:
                      min_unit = min_unit + impact
                bounds[i,j,0] = max(max_unit,1)
                bounds[i,j,1] = max(-min_unit,1)

            # Create a new model
            model = Model("mip1")

            if (not(disp_opt)):
                # Donot display output solving
                # https://stackoverflow.com/a/37137612
                model.params.outputflag = 0

            # Create variables
            g    = model.addVars(lst, lb=-GRB.INFINITY, name="g")
            h    = model.addVars(lst, lb=0.0          , name="h")
            hbar = model.addVars(lst, lb=0.0          , name="hbar")
            z    = model.addVars(lst, vtype=GRB.BINARY, name="z")

            p = model.addVars(p_lst, name="p")
            q = model.addVars(q_lst, name="q")

            # Specify bounds of the input variables.
            # 0 index in bounds is for maxima and 1 index in bounds is for minima
            model.addConstrs( h[0, k] <= bounds[0, k, 0] for k in range(nodes_per_layer[0]))
            model.addConstrs( h[0, k] >= bounds[0, k, 1] for k in range(nodes_per_layer[0]))





            # Add constraints for all the nodes starting from the input layer till nodes in this layer
            for m in range(1, run_till_layer_index):
                for n in range(nodes_per_layer[m]):

                    name = "c_" + str(m) + str(n)
                    model.addConstr(quicksum(weights[m-1][n,k] * h[m-1,k] for k in range(nodes_per_layer[m-1])) + bias[m-1][n] - g[m,n] == 0, name + "_1")
                    model.addConstr(g[m,n]    == h[m,n] - hbar[m,n], name + "_2")
                    model.addConstr(h[m,n]    <= 2*bounds[m, n, 0] * z[m,n], name + "_3")

                    model.addConstr(hbar[m,n] <= 2*bounds[m, n, 1] * ( 1 - z[m,n] ), name + "_4")

            for (m,n) in p_lst:
                    model.addConstr(p[m,n] <= z[m,n])
            for (m,n) in q_lst:
                    model.addConstr(q[m,n] <= 1-z[m,n])

            # model.setObjective(quicksum(p[m,n]+q[m,n] for m in range(1, run_till_layer_index) for n in range(nodes_per_layer[m])) , GRB.MAXIMIZE)
            model.setObjective(quicksum(p[m,n] for (m,n) in p_lst) +  quicksum(q[m,n] for (m,n) in q_lst) , GRB.MAXIMIZE)

   
            try:
                    print("SOLVING FOR",network)
                    positive_units = set()
                    negative_units = set()
                    model.params.LazyConstraints = 1
                    model.params.StartNodeLimit = 1
                    if args.time_limit != None:
                        model.setParam(GRB.Param.TimeLimit, args.time_limit-time.time()+time_before)
                    model.optimize(networkcallback) 
                    if args.time_limit != None and time.time()-time_before > args.time_limit:
                        timed_out = True
            except GurobiError as e:
                    print("3 Error reported")

            for m in range(1, run_till_layer_index):
              for n in range(nodes_per_layer[m]):
                #if (m,n) in positive_units and not (m,n) in negative_units:
                if (m,n) in q_lst and not (m,n) in negative_units:
                    stably_active[m].append(n)
                #elif (m,n) in negative_units and not (m,n) in positive_units:
                elif (m,n) in p_lst and not (m,n) in positive_units:
                    stably_inactive[m].append(n)    

              if not timed_out:
                print("Layer %d Completed..." %(m))

                matrix_list = []
                #for j in stably_active[m]:
                #  matrix_list.append([weights[m-1][j,k] for k in range(nodes_per_layer[m-1])])
                #  import pdb;pdb.set_trace()
                matrix_list = [weights[m-1][j,:] for j in stably_active[m]]
                print("Active: ", stably_active[m])
                rank = np.linalg.matrix_rank(np.array(matrix_list).astype(np.float64))
                #import pdb;pdb.set_trace()
                #print("Active rank: ", rank, "out of", len(stably_active[m]))
                print("Inactive: ", stably_inactive[m])
                f.write(str(len(stably_active[m]))+", "+str(rank)+", "+str(len(stably_inactive[m]))+",, ")
              else:
                f.write("-, -, -,, ")

# Print the maxima and the minima from first hidden layer to output layer

time_after = time.time()
f.write(str(time_after-time_before)+",, ")
f.write(args.formulation+", "+args.feasible+",, "+str(remaining)+",, \n")
f.close()
np.save(stable_neurons_path, {'stably_active': stably_active, 'stably_inactive': stably_inactive})
#print_bounds(tot_layers, nodes_per_layer, bounds)


################################################################################
# If we have to do only bounds, do not go for activations
################################################################################
if (args.bounds_only_flag):
    sys.exit()


# Reseting the parameters of the model
model.reset()


################################################################################
# Custom callback function
# Termination is normally handled through Gurobi parameters
# (MIPGap, NodeLimit, etc.).  You should only use a callback for
# termination if the available parameters don't capture your desired
# termination criterion.
#
# Reference:
# https://www.gurobi.com/documentation/8.1/examples/callback_py.html
################################################################################
def mycallback(model, where):
    # General MIP callback
    if where == GRB.Callback.MIP:
        obj_bnd = model.cbGet(GRB.Callback.MIP_OBJBND)
        if (obj_bnd < 0):
            print("Objective bound (Soln of relaxed LP) < 0")
            model.terminate()

    # If an MIP solution is found
    elif where == GRB.Callback.MIPSOL:
        model._sol_count += 1
        obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)

        #print("\nObjective f = %f" %(obj_val))

        if (obj_val > 0):
            model._val_sol_count += 1

            if (model._val_sol_count % print_freq == 0):
                sys.stdout.write("Valid/Total Solutions %d / %d Time %d s\n" %(model._val_sol_count, model._sol_count, int(time.time() - now) ))
                sys.stdout.flush()

            #print ("g = ")
            #print (model.cbGetSolution(g))
            #print ("h = ")
            #print (model.cbGetSolution(h))
            #print ("hBar = ")
            #print (model.cbGetSolution(hbar))
            #print ("Binary Variables z = ")
            #print (model.cbGetSolution(z))

            # We want to remove the solution that we just found, so that it does
            # not repeat and also to avoid the solver from finishing before
            # enumerating all positive solutions because it found a solution
            # that provably maximizes f
            #
            # The current solution will have
            # Sum of variables which are 1 - Sum of variables which are 0 = #Variables that are 1
            # So, we add a lazy cut
            # Sum of variables which are 1 - Sum of variables which are 0 <= #Variables that are 1 - 1

            # https://groups.google.com/forum/#!topic/gurobi/d38iycxUIps
            vals = model.cbGetSolution(z)
            expr = LinExpr(0.0)
            ones_cnt = 0

            line_to_write = ""

            # No need for input activations
            for m in range(1, run_till_layer_index):
                for n in range(nodes_per_layer[m]):
                    if (vals[m, n] > 0.9 ):
                        expr.add(model._z[m, n], 1.0)
                        ones_cnt += 1
                        term = "1 "
                    else:
                        expr.add(model._z[m, n], -1.0)
                        term = "0 "

                    line_to_write += term

            line_to_write += "\n"
            # Write the line to the file
            model._my_file.write(line_to_write)
            if (show_activations):
                sys.stdout.write(line_to_write)
                sys.stdout.flush()

            # Add a lazy constraint so that this solution does not appear again
            constraint = model.cbLazy(expr <= ones_cnt-1)
            # print("Ones_cnt = %d" %(ones_cnt))
            # print(expr)
        else:
            pass
            # print("Invalid Solution Found")

print("")

# Writing the activation patterns to a file
my_file = open(os.path.join(os.path.dirname(args.input), activation_pattern_file), "w")
my_file.write("n = [" + ', '.join(map(str, nodes_per_layer)) + "]\n")


################################################################################
# Find all possible activation patterns
################################################################################
# Create a new model
model = Model("mip2")

if (not(disp_opt)):
    # Donot display output solving
    # https://stackoverflow.com/a/37137612
    model.params.outputflag = 0

# Create variables
g    = model.addVars(lst, lb=-GRB.INFINITY, name="g")
h    = model.addVars(lst, lb=0.0          , name="h")
hbar = model.addVars(lst, lb=0.0          , name="hbar")
z    = model.addVars(lst, vtype=GRB.BINARY, name="z")
f    = model.addVar (     lb=0.0          , name="f")

# For Lazy cuts
# Set the "LazyConstraints" parameter to 1 in order to tell Gurobi that it does
# not know all the model constraints. Essentially, this disables dual presolve
# reductions.
model.params.LazyConstraints = 1

# Specify bounds of the input variables.
# 0 index in bounds is for maxima and 1 index in bounds is for minima
model.addConstrs( h[0, k] <= bounds[0, k, 0] for k in range(nodes_per_layer[0]))
model.addConstrs( h[0, k] >= bounds[0, k, 1] for k in range(nodes_per_layer[0]))


# Add constraints for all the nodes starting from the input layer till the last layer
for m in range(1, run_till_layer_index):
    for n in range(nodes_per_layer[m]):
        name = "c_" + str(m) + str(n)
        model.addConstr(quicksum(weights[m-1][n,k] * h[m-1,k] for k in range(nodes_per_layer[m-1])) + bias[m-1][n] - g[m,n] == 0, name + "_1")
        model.addConstr(g[m,n]    == h[m,n] - hbar[m,n], name + "_2")
        model.addConstr(h[m,n]    <= bounds[m, n, 0] * z[m,n], name + "_3")
        model.addConstr(hbar[m,n] <= bounds[m, n, 1] * ( 1 - z[m,n] ), name + "_4")

        # For stably inactive nodes, we add bounding constraint with bounds as 1
        if (bounds[m, n, 0] > 0):
            model.addConstr(f         <= h[m,n] + ( 1 - z[m,n] ) * bounds[m, n, 0] , name + "_10")
        else:
            model.addConstr(f         <= h[m,n] + ( 1 - z[m,n] )                   , name + "_10")


model.setObjective(f , GRB.MAXIMIZE)

# Pass data into mycallback function
model._z               = z
model._sol_count       = 0
model._val_sol_count   = 0
model._nodes_per_layer = nodes_per_layer
model._my_file         = my_file

now = time.time()

if (show_activations):
    print("Activation Patterns ")

try:
    model.optimize(mycallback)
except GurobiError:
    print("3 Error reported")

# close the activations pattern file
my_file.close()

print("------------------------------------------------------------------------")
print("%d Valid Activations Found. Time %.3f s"%(model._val_sol_count, (time.time() - now) ))
print("------------------------------------------------------------------------")
