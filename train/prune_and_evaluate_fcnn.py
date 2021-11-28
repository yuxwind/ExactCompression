#!/usr/bin/python3

"""
    Prunes the network and then sees if after pruning the performace is exactly the same
    
    For pruning, it takes a file named "inactive.dat" in the same folder as the weights file
    which contains the indices of the layer and indices of the nodes separated by space.
 
    Sample Run -
    python3 prune_and_evaluate_fcnn.py --arch fcnn2h --resume models/fcnn_run_61/checkpoint_120.tar -e
"""

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import fcnn
#import VGG
import torch.nn.functional as F
import numpy as np

model_names = sorted(name for name in fcnn.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("fcnn")
                     and callable(fcnn.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument( '--arch', metavar='ARCH', default='fcnn1',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: fcnn1)')
parser.add_argument('-j', '--workers'         , type=int  , default=4     , metavar='N'         , help='number of data loading workers (default: 4)')
parser.add_argument(      '--start-epoch'     , type=int  , default=0     , metavar='N'         , help='manual epoch number (useful on restarts)')
parser.add_argument(      '--test-batch-size' , type=int  , default=10000 , metavar='T'         , help='input batch size for testing (default: 10000)')
parser.add_argument(      '--print-freq'      , type=int  , default=1     , metavar='N'         , help='print frequency (default: 1)')

# Evaluation options
parser.add_argument(      '--resume'          , type=str  , default=''    , metavar='PATH'      , help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate'        , dest='evaluate'           , action='store_true' , help='evaluate model on validation set')
parser.add_argument(      '--half'            , dest='half'               , action='store_true' , help='use half-precision(16-bit) ')

# More advanced options
parser.add_argument('-s', '--std_dev'         , type=float, default=0.05                         , help='if use augmentation, then standard value of noise (default=0.05)')
parser.add_argument('-a', '--augmentation'    , dest='augmentation'       , action='store_true'  , help='use augmentation (default=False)')  
parser.add_argument('-v', '--adversarial'     , dest='adversarial'        , action='store_true'  , help='use adversarial attacks (default=False)')   

best_prec1 = 0
dataset    = "MNIST"

################################################################################
# main function
################################################################################
def main():
    global args, best_prec1
    args = parser.parse_args()

    cudnn.benchmark = True
    use_cuda        = True
    device          = torch.device("cuda" if use_cuda else "cpu")    
    fcnn_flag       = True
    inactive_file   = "inactive_input_0_1.dat" 

    model = fcnn.__dict__[args.arch]()
    print(model)
    
    model.features = torch.nn.DataParallel(model.features)
    model.to(device)
        
    print("*************************************************************************************************");
    print("*************************************************************************************************");
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    
    stddev = args.std_dev
    distbn = torch.distributions.normal.Normal(0, stddev)

    
    if(dataset == "CIFAR10"):
        print("Running on CIFAR10")
        input_dim = 1024
        #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        normalize = transforms.Normalize(mean=[0], std=[1])
    elif (dataset == "MNIST"):
        print("Running on MNIST")
        input_dim = 784
        normalize = transforms.Normalize(mean=[0], std=[1]) #Images are already loaded in [0,1]
    else:
        print("Unknown Dataset")
        

    transform_list=[transforms.ToTensor(), normalize]
    
    if args.evaluate:
        if args.adversarial: 
            pass    
        else:
            if args.augmentation:
                print("Using augmentation. Std deviation of the noise while testing/evaluation = " + str(stddev))
                transform_list.append(transforms.Lambda(lambda img: img + distbn.sample(img.shape))) 
            else:
                print("No augmentation used in testing")
            
            if(dataset == "CIFAR10"):
                transform_list=[transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), normalize]    
                val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose(transform_list), download=True),
                batch_size=args.test_batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)

            elif (dataset == "MNIST"):
                val_loader = torch.utils.data.DataLoader(
                datasets.MNIST(root='./data', train=False, transform=transforms.Compose(transform_list), download=True),
                batch_size=args.test_batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)                               
            else:
                print("Unknown Dataset")
            
            
            # Before Pruning
            print("\n\n-----------------------------------------------------------")
            print("               Before Pruning")
            print("-----------------------------------------------------------")
            acc = validate(val_loader, model, criterion, 1, device, fcnn)


            # In Pruning, we read the file "inactive.dat" and set the weights and biases for the
            # corresponding units to be zero
            inactive_units      = np.genfromtxt( os.path.join(os.path.dirname(args.resume), inactive_file), delimiter=' ').astype(int)
            # sort in increasing index            
            #inactive_units      = inactive_units[inactive_units[:,0].argsort()]

            print("\n\n")
            #print("-----------------------------------------------------------")
            print("Pruning %d inactive nodes" %(inactive_units.shape[0]))
            #print("-----------------------------------------------------------")            
            uniq_layers_index = np.unique(inactive_units[:, 0])

            wt_layers = 0
            bias_layers = 0

            params = model.state_dict()

            for key, value in params.items():
                if ('weight' in  key):
                    wt_layers += 1

                    if (wt_layers in uniq_layers_index):                      
                        indices = inactive_units[ inactive_units[:,0] == wt_layers, 1]
                        for i in range(indices.shape[0]):
                            value[indices[i],:] = 0

                if ('bias' in key):
                    bias_layers += 1
                    if (bias_layers in uniq_layers_index):
                        indices = inactive_units[ inactive_units[:,0] == bias_layers, 1]
                          
                        for i in range(indices.shape[0]):
                            value[indices[i]] = 0
                    
            #print("-----------------------------------------------------------")
            print("Print values to double check if the weight and bias values for pruned nodes are zero")
            params = model.state_dict()
            wt_layers = 0
            bias_layers = 0
            for key, value in params.items():
                if ('weight' in  key):
                    wt_layers += 1

                    if (wt_layers in uniq_layers_index):
                        print(value.shape)
                        indices = inactive_units[ inactive_units[:,0] == wt_layers, 1]
                        
                        for i in range(indices.shape[0]):
                            pass
                            #print(value[indices[i],:])

                if ('bias' in key):
                    bias_layers += 1
                    if (bias_layers in uniq_layers_index):
                        indices = inactive_units[ inactive_units[:,0] == bias_layers, 1]
                          
                        for i in range(indices.shape[0]):
                            pass
                            #print(value[indices[i]])


            print("\n\n-----------------------------------------------------------")
            print("               After Pruning")
            print("-----------------------------------------------------------")
            acc = validate(val_loader, model, criterion, 1, device, fcnn)                    
            return

################################################################################
# Run evaluation
################################################################################
def validate(val_loader, model, criterion, epoch, device, fcnn):

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (data, target) in enumerate(val_loader):
        
        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        if args.half:
            data = data.half()

        if fcnn:           
            data = data.view(data.shape[0],-1)
            
        # compute output
        output,_ = model(data)
        loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data , target)[0]
        losses.update   (loss.item() , data.size(0))
        top1.update     (prec1.item(), data.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print('Epoch: [{0}] * Prec@1 {top1.avg:.3f}'
          .format(epoch, top1=top1))

    return top1.avg

"""Computes and stores the average and current value"""    
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


################################################################################
# Computes the precision@k for the specified values of k
################################################################################
def accuracy(output, target, topk=(1,)):

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
