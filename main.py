##################################################
# Main script used to commence experiments
##################################################
# Author: Marius Bock
# Email: marius.bock(at)uni-siegen.de
##################################################

import argparse
import datetime
import json
import os
import time
import sys

from data_processing.preprocess_data import load_dataset
from model.validation import cross_participant_cv, train_valid_split

from misc.logging import Logger
from misc.torchutils import seed_torch
import neptune

"""
DATASET OPTIONS:
- TEST_TYPE: type of experiment commenced (see data preprocess_data.py for insights)
- TEST_CASE: type of test case conducted (see data preprocess_data.py for insights)
- SW_LENGTH: length of sliding window
- SW_UNIT: unit in which length of sliding window is measured
- SW_OVERLAP: overlap ratio between sliding windows (in percent, i.e. 60 = 60%)
- INCLUDE_VOID: boolean whether to include void class in datasets
"""

TEST_TYPE = 'hypertuning'
TEST_CASE = 'loso_us'
SW_LENGTH = 1
SW_UNIT = 'seconds'
SW_OVERLAP = 60
INCLUDE_VOID = False

"""
NETWORK OPTIONS:
- NETWORK: network architecture to be used (e.g. 'deepconvlstm')
- LSTM: boolean whether to employ a lstm after convolution layers
- NB_UNITS_LSTM: number of hidden units in each LSTM layer
- NB_LAYERS_LSTM: number of layers in LSTM
- CONV_BLOCK_TYPE: type of convolution blocks employed ('normal', 'skip' or 'fixup')
- NB_CONV_BLOCKS: number of convolution blocks employed
- NB_FILTERS: number of convolution filters employed in each layer of convolution blocks
- FILTER_WIDTH: width of convolution filters (e.g. 11 = 11x1 filter)
- DILATION: dilation factor employed on convolutions (set 1 for not dilation)
- DROP_PROB: dropout probability in dropout layers
- POOLING: boolean whether to employ a pooling layer after convolution layers
- BATCH_NORM: boolean whether to apply batch normalisation in convolution blocks
- REDUCE_LAYER: boolean whether to employ a reduce layer after convolution layers
- POOL_TYPE: type of pooling employed in pooling layer
- POOL_KERNEL_WIDTH: width of pooling kernel (e.g. 2 = 2x1 pooling kernel)
- REDUCE_LAYER_OUTPUT: size of the output after the reduce layer (i.e. what reduction is to be applied) 
"""

NETWORK = 'deepconvlstm'
NO_LSTM = False
NB_UNITS_LSTM = 128
NB_LAYERS_LSTM = 1
CONV_BLOCK_TYPE = 'normal'
NB_CONV_BLOCKS = 2
NB_FILTERS = 64
FILTER_WIDTH = 11
DILATION = 1
DROP_PROB = 0.5
POOLING = False
BATCH_NORM = False
REDUCE_LAYER = False
POOL_TYPE = 'max'
POOL_KERNEL_WIDTH = 2
REDUCE_LAYER_OUTPUT = 8

"""
TRAINING OPTIONS:
- SEED: random seed which is to be employed
- VALID_TYPE: (cross-)validation type; either 'cross-participant', 'split' or 'kfold'
- VALID_EPOCH: which epoch used for evaluation; either 'best' or 'last'
- BATCH_SIZE: size of the batches
- EPOCHS: number of epochs during training
- OPTIMIZER: optimizer to use; either 'rmsprop', 'adadelta' or 'adam'
- LR: learning rate to employ for optimizer
- WEIGHT_DECAY: weight decay to employ for optimizer
- WEIGHTS_INIT: weight initialization method to use to initialize network
- LOSS: loss to use ('cross_entropy', 'maxup')
- SMOOTHING: degree of label smoothing employed if cross-entropy used
- GPU: name of GPU to use (e.g. 'cuda:0')
- WEIGHTED: boolean whether to use weighted loss calculation based on support of each class
- SHUFFLING: boolean whether to use shuffling during training
- ADJ_LR: boolean whether to adjust learning rate if no improvement
- LR_SCHEDULER: type of learning rate scheduler to employ ('step_lr', 'reduce_lr_on_plateau')
- LR_STEP: step size of learning rate scheduler (patience if plateau).
- LR_DECAY: decay factor of learning rate scheduler.
- EARLY_STOPPING: boolean whether to stop the network training early if no improvement 
- ES_PATIENCE: patience (i.e. number of epochs) after which network training is stopped if no improvement
"""

SEED = 1
VALID_TYPE = 'cross-participant'
VALID_EPOCH = 'best'
BATCH_SIZE = 100
EPOCHS = 30
OPTIMIZER = 'adam'
LR = 1e-4
WEIGHT_DECAY = 1e-6
WEIGHTS_INIT = 'xavier_normal'
LOSS = 'cross_entropy'
SMOOTHING = 0.0
GPU = 'cuda:0'
WEIGHTED = False
SHUFFLING = False
ADJ_LR = False
LR_SCHEDULER = 'step_lr'
LR_STEP = 10
LR_DECAY = 0.9
EARLY_STOPPING = False
ES_PATIENCE = 10

"""
LOGGING OPTIONS:
- NAME: name of the experiment; used for logging purposes
- NEPTUNE: boolean whether to use neptune.ai for logging (please provide credentials below!)
- VERBOSE: boolean whether to print batchwise results during epochs
- PRINT_FREQ: number of batches after which batchwise results are printed
- SAVE_PREDICTIONS: boolean whether to save predictions made by models
- SAVE_MODEL: boolean whether to save the model after last epoch as a checkpoint file
- SAVE_ANALYSIS: boolean whether to save analysis dataframe, i.e. csv containing all scores
"""

NAME = 'test_experiment'
NEPTUNE = False
VERBOSE = False
PRINT_FREQ = 100
SAVE_PREDICTIONS = False
SAVE_CHECKPOINTS = False
SAVE_ANALYSIS = False


def main(args):
    if args.neptune:
        run = neptune.init_run(
        project=None,
        api_token=None,
        )
    else:
        run = None

    ts = datetime.datetime.fromtimestamp(int(time.time()))
    log_dir = os.path.join('logs', args.test_type, args.test_case, args.network, str(ts))
    sys.stdout = Logger(os.path.join(log_dir, 'log.txt'))

    # save the current cfg
    with open(os.path.join(log_dir, 'cfg.txt'), 'w') as fid:
        json.dump(args.__dict__, fid, indent=2)
    
    if args.neptune:
        run['config'].upload(os.path.join(log_dir, 'cfg.txt'))
        run['params'] = args
    
    # apply the chosen random seed to all relevant parts
    seed_torch(args.seed)
        
    

    ################################################## DATA LOADING ####################################################

    print('Loading data...')
    train, valid, subjects, nb_classes, class_names, sampling_rate, has_void = \
        load_dataset(test_type=args.test_type, test_case=args.test_case, include_void=args.include_void)

    args.subjects = subjects
    args.sampling_rate = sampling_rate
    args.nb_classes = nb_classes
    args.class_names = class_names
    args.has_void = has_void

    ############################################# TRAINING #############################################################

    if valid is None:
        print("LOSO dataset with size: | {0} |".format(train.shape))
    else:
        print("Split datasets with size: | train {0} | valid {1} |".format(train.shape, valid.shape))

    if valid is None:
        _ = cross_participant_cv(train, args, log_dir, run)
    else:
        _ = train_valid_split(train, valid, args, log_dir, run)

    print("\nALL FINISHED")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # DATASET OPTIONS
    parser.add_argument('--print_freq', default=PRINT_FREQ, type=int)
    parser.add_argument('--test_type', default=TEST_TYPE, type=str)
    parser.add_argument('--test_case', default=TEST_CASE, type=str)
    parser.add_argument('--sw_length', default=SW_LENGTH, type=float)
    parser.add_argument('--sw_unit', default=SW_UNIT, type=str)
    parser.add_argument('--sw_overlap', default=SW_OVERLAP, type=int)
    parser.add_argument('--include_void', default=INCLUDE_VOID, action='store_true')

    # NETWORK OPTIONS
    parser.add_argument('--network', default=NETWORK, type=str)
    parser.add_argument('--no_lstm', default=NO_LSTM, action='store_true')
    parser.add_argument('--nb_units_lstm', default=NB_UNITS_LSTM, type=int)
    parser.add_argument('--nb_layers_lstm', default=NB_LAYERS_LSTM, type=int)
    parser.add_argument('--conv_block_type', default=CONV_BLOCK_TYPE, type=str)
    parser.add_argument('--nb_conv_blocks', default=NB_CONV_BLOCKS, type=int)
    parser.add_argument('--nb_filters', default=NB_FILTERS, type=int)
    parser.add_argument('--filter_width', default=FILTER_WIDTH, type=int)
    parser.add_argument('--dilation', default=DILATION, type=int)
    parser.add_argument('--drop_prob', default=DROP_PROB, type=float)
    parser.add_argument('--pooling', default=POOLING, action='store_true')
    parser.add_argument('--batch_norm', default=BATCH_NORM, action='store_true')
    parser.add_argument('--reduce_layer', default=REDUCE_LAYER, action='store_true')
    parser.add_argument('--pool_type', default=POOL_TYPE, type=str)
    parser.add_argument('--pool_kernel_width', default=POOL_KERNEL_WIDTH, type=int)
    parser.add_argument('--reduce_layer_output', default=REDUCE_LAYER_OUTPUT, type=int)

    # TRAINING OPTIONS
    parser.add_argument('--seed', default=SEED, type=int)
    parser.add_argument('--valid_type', default=VALID_TYPE, type=str)
    parser.add_argument('--valid_epoch', default=VALID_EPOCH, type=str)
    parser.add_argument('--batch_size', default=BATCH_SIZE, type=int)
    parser.add_argument('--epochs', default=EPOCHS, type=int)
    parser.add_argument('--optimizer', default=OPTIMIZER, type=str)
    parser.add_argument('--learning_rate', default=LR, type=float)
    parser.add_argument('--weight_decay', default=WEIGHT_DECAY, type=float)
    parser.add_argument('--weights_init', default=WEIGHTS_INIT, type=str)
    parser.add_argument('--loss', default=LOSS, type=str)
    parser.add_argument('--smoothing', default=SMOOTHING, type=float)
    parser.add_argument('--gpu', default=GPU, type=str)
    parser.add_argument('--weighted', default=WEIGHTED, action='store_true')
    parser.add_argument('--shuffling', default=WEIGHTED, action='store_true')
    parser.add_argument('--adj_lr', default=ADJ_LR, action='store_true')
    parser.add_argument('--lr_scheduler', default=LR_SCHEDULER, type=str)
    parser.add_argument('--lr_step', default=LR_STEP, type=int)
    parser.add_argument('--lr_decay', default=LR_DECAY, type=float)
    parser.add_argument('--early_stopping', default=EARLY_STOPPING, action='store_true')
    parser.add_argument('--es_patience', default=ES_PATIENCE, type=int)

    # LOGGING OPTIONS
    parser.add_argument('--name', default=NAME, type=str)
    parser.add_argument('--neptune', default=NEPTUNE, action='store_true')
    parser.add_argument('--verbose', default=VERBOSE, action='store_true')
    parser.add_argument('--save_predictions', default=SAVE_CHECKPOINTS, action='store_true')
    parser.add_argument('--save_checkpoints', default=SAVE_CHECKPOINTS, action='store_true')
    parser.add_argument('--save_analysis', default=SAVE_ANALYSIS, action='store_true')

    args = parser.parse_args()

    main(args)
