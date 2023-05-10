##################################################
# All functions related to validating a model
##################################################
# Author: Marius Bock
# Email: marius.bock(at)uni-siegen.de
##################################################

import os
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_score, recall_score, f1_score

from data_processing.sliding_window import apply_sliding_window
from misc.osutils import mkdir_if_missing
from model.AttendAndDiscriminate import AttendAndDiscriminate
from model.DeepConvLSTM import DeepConvLSTM
from model.train import train, init_optimizer, init_loss, init_scheduler
from neptune.types import File

def cross_participant_cv(data, args, log_dir=None, run=None):
    """
    Method to apply cross-participant cross-validation (also known as leave-one-subject-out cross-validation).

    :param data: numpy array
        Data used for applying cross-validation
    :param args: dict
        Args object containing all relevant hyperparameters and settings
    :param log_dir: string
        Logging directory
    :return pytorch model
        Trained network
    """
    print('\nCALCULATING CROSS-PARTICIPANT SCORES USING LOSO CV.\n')
    cp_scores = np.zeros((4, args.nb_classes, int(np.max(data[:, 0]) + 1)))
    train_val_gap = np.zeros((4, int(np.max(data[:, 0]) + 1)))
    all_train_output = None
    all_val_output = None
    orig_lr = args.learning_rate

    for i, sbj in enumerate(np.unique(data[:, 0])):
        print('\n VALIDATING FOR SUBJECT {0}; {1} OF {2}'.format(args.subjects[int(sbj)], int(sbj) + 1, int(len(np.unique(data[:, 0])))))
        train_data = data[data[:, 0] != sbj]
        val_data = data[data[:, 0] == sbj]
        args.learning_rate = orig_lr

        # Sensor data is segmented using a sliding window mechanism
        X_train, y_train = apply_sliding_window(train_data[:, :-1], train_data[:, -1],
                                                sliding_window_size=args.sw_length,
                                                unit=args.sw_unit,
                                                sampling_rate=args.sampling_rate,
                                                sliding_window_overlap=args.sw_overlap,
                                                )

        X_val, y_val = apply_sliding_window(val_data[:, :-1], val_data[:, -1],
                                            sliding_window_size=args.sw_length,
                                            unit=args.sw_unit,
                                            sampling_rate=args.sampling_rate,
                                            sliding_window_overlap=args.sw_overlap,
                                            )

        X_train, X_val = X_train[:, :, 1:], X_val[:, :, 1:]

        args.window_size = X_train.shape[1]
        args.nb_channels = X_train.shape[2]

        # network initialization
        if args.network == 'deepconvlstm':
            net = DeepConvLSTM(config=vars(args))
        elif args.network == 'attendanddiscriminate':        
            net = AttendAndDiscriminate(args.nb_channels, args.nb_classes, args.nb_units_lstm, args.nb_filters, args.filter_width, args.nb_layers_lstm, False, args.drop_prob, 0.5, 0.5, 'ReLU', 1, args.gpu, args.weights_init)
        else:
            print("Did not provide a valid network name!")

        # optimizer initialization
        opt = init_optimizer(net, args)

        # optimizer initialization
        loss = init_loss(args)

        # lr scheduler initialization
        if args.adj_lr:
            print('Adjusting learning rate according to scheduler: ' + args.lr_scheduler)
            scheduler = init_scheduler(opt, args)
        else:
            scheduler = None

        net, checkpoint, val_output, train_output = train(X_train, y_train, X_val, y_val,
                                                          network=net, optimizer=opt, loss=loss, lr_scheduler=scheduler,
                                                          config=vars(args), run=run, name='sbj_' + str(int(sbj))
                                                          )

        if args.save_checkpoints:
            print('Saving checkpoint...')
            if args.valid_epoch == 'last':
                c_name = os.path.join(log_dir, "checkpoint_last_{}_{}.pth".format(args.subjects[int(sbj)], str(args.name)))
            else:
                c_name = os.path.join(log_dir, "checkpoint_best_{}_{}.pth".format(args.subjects[int(sbj)], str(args.name)))
            torch.save(checkpoint, c_name)

        if args.save_predictions:
            print('Saving predictions...')
            if args.valid_epoch == 'last':
                p_name = os.path.join(log_dir, "predictions_last_{}_{}.csv".format(args.subjects[int(sbj)], str(args.name)))
            else:
                p_name = os.path.join(log_dir, "predictions_best_{}_{}.csv".format(args.subjects[int(sbj)], str(args.name)))
            pd.DataFrame(val_output).to_csv(p_name)

        if all_val_output is None:
            all_train_output = train_output
            all_val_output = val_output
        else:
            all_train_output = np.concatenate((all_train_output, train_output), axis=0)
            all_val_output = np.concatenate((all_val_output, val_output), axis=0)

        # fill values for normal evaluation
        labels = list(range(0, args.nb_classes))
        t_conf_mat = confusion_matrix(train_output[:, 1], train_output[:, 0], normalize='true', labels=labels)
        t_acc = t_conf_mat.diagonal() / t_conf_mat.sum(axis=1)
        t_prec = precision_score(train_output[:, 1], train_output[:, 0], average=None, zero_division=1, labels=labels)
        t_rec = recall_score(train_output[:, 1], train_output[:, 0], average=None, zero_division=1, labels=labels)
        t_f1 = f1_score(train_output[:, 1], train_output[:, 0], average=None, zero_division=1, labels=labels)
        
        v_conf_mat = confusion_matrix(val_output[:, 1], val_output[:, 0], normalize='true', labels=labels)
        v_acc = v_conf_mat.diagonal()/v_conf_mat.sum(axis=1)
        v_prec = precision_score(val_output[:, 1], val_output[:, 0], average=None, zero_division=1, labels=labels)
        v_rec = recall_score(val_output[:, 1], val_output[:, 0], average=None, zero_division=1, labels=labels)
        v_f1 = f1_score(val_output[:, 1], val_output[:, 0], average=None, zero_division=1, labels=labels)
        
        cp_scores[0, :, int(sbj)] = v_acc
        cp_scores[1, :, int(sbj)] = v_prec
        cp_scores[2, :, int(sbj)] = v_rec
        cp_scores[3, :, int(sbj)] = v_f1

        # fill values for train val gap evaluation
        train_val_gap[0, int(sbj)] = np.nanmean(t_acc) - np.nanmean(v_acc)
        train_val_gap[1, int(sbj)] = np.nanmean(t_prec) - np.nanmean(v_prec)
        train_val_gap[2, int(sbj)] = np.nanmean(t_rec) - np.nanmean(v_rec)
        train_val_gap[3, int(sbj)] = np.nanmean(t_f1) - np.nanmean(v_f1)

        print("SUBJECT {0} VALIDATION RESULTS: ".format(args.subjects[int(sbj)]))
        print("Accuracy: {:>4.2f} (%)".format(np.nanmean(v_acc) * 100))
        print("Precision: {:>4.2f} (%)".format(np.nanmean(v_prec) * 100))
        print("Recall: {:>4.2f} (%)".format(np.nanmean(v_rec) * 100))
        print("F1: {:>4.2f} (%)".format(np.nanmean(v_f1) * 100))
        
         # save per-subject confusion matrix
        _, ax = plt.subplots(figsize=(15, 15), layout="constrained")
        ax.set_title('Confusion Matrix Subject ' + str(int(sbj)))
        conf_disp = ConfusionMatrixDisplay(confusion_matrix=v_conf_mat, display_labels=args.class_names)    
        conf_disp.plot(ax=ax, xticks_rotation='vertical', colorbar=False)
        mkdir_if_missing(os.path.join(log_dir, 'conf_mats'))
        plt.savefig(os.path.join(log_dir, 'conf_mats', 'sbj_' + str(int(sbj)) + '.png'))
        if run is not None:
            run['conf_matrices'].append(File(os.path.join(log_dir, 'conf_mats', 'sbj_' + str(int(sbj)) + '.png')), name='sbj_' + str(int(sbj)))

    if args.save_analysis:
        cp_score_acc = pd.DataFrame(cp_scores[0, :, :], index=None)
        cp_score_acc.index = args.class_names
        cp_score_acc.columns = args.subjects
        cp_score_prec = pd.DataFrame(cp_scores[1, :, :], index=None)
        cp_score_prec.index = args.class_names
        cp_score_prec.columns = args.subjects
        cp_score_rec = pd.DataFrame(cp_scores[2, :, :], index=None)
        cp_score_rec.index = args.class_names
        cp_score_rec.columns = args.subjects
        cp_score_f1 = pd.DataFrame(cp_scores[3, :, :], index=None)
        cp_score_f1.index = args.class_names
        cp_score_f1.columns = args.subjects
        tv_gap = pd.DataFrame(train_val_gap, index=None)
        tv_gap.index = ['accuracy', 'precision', 'recall', 'f1']
        tv_gap.columns = args.subjects
        
        cp_score_acc.to_csv(os.path.join(log_dir, 'cp_scores_acc_{}.csv'.format(args.name)))
        cp_score_prec.to_csv(os.path.join(log_dir, 'cp_scores_prec_{}.csv').format(args.name))
        cp_score_rec.to_csv(os.path.join(log_dir, 'cp_scores_rec_{}.csv').format(args.name))
        cp_score_f1.to_csv(os.path.join(log_dir, 'cp_scores_f1_{}.csv').format(args.name))
        tv_gap.to_csv(os.path.join(log_dir, 'train_val_gap_{}.csv').format(args.name))
        
        if run is not None:    
            run["analysis"].upload(os.path.join(log_dir, 'cp_scores_acc_{}.csv'.format(args.name)))
            run["analysis"].upload(os.path.join(log_dir, 'cp_scores_prec_{}.csv'.format(args.name)))
            run["analysis"].upload(os.path.join(log_dir, 'cp_scores_rec_{}.csv'.format(args.name)))
            run["analysis"].upload(os.path.join(log_dir, 'cp_scores_f1_{}.csv'.format(args.name)))
            run["analysis"].upload(os.path.join(log_dir, 'train_val_gap_{}.csv'.format(args.name)))

    # fill values for normal evaluation
    labels = list(range(0, args.nb_classes))
    t_conf_mat = confusion_matrix(all_train_output[:, 1], all_train_output[:, 0], normalize='true', labels=labels)
    t_acc = t_conf_mat.diagonal()/t_conf_mat.sum(axis=1)
    t_prec = precision_score(all_train_output[:, 1], all_train_output[:, 0], average=None, zero_division=1, labels=labels)
    t_rec = recall_score(all_train_output[:, 1], all_train_output[:, 0], average=None, zero_division=1, labels=labels)
    t_f1 = f1_score(all_train_output[:, 1], all_train_output[:, 0], average=None, zero_division=1, labels=labels)
        
    v_conf_mat = confusion_matrix(all_val_output[:, 1], all_val_output[:, 0], normalize='true', labels=labels)
    v_acc = v_conf_mat.diagonal()/v_conf_mat.sum(axis=1)
    v_prec = precision_score(all_val_output[:, 1], all_val_output[:, 0], average=None, zero_division=1, labels=labels)
    v_rec = recall_score(all_val_output[:, 1], all_val_output[:, 0], average=None, zero_division=1, labels=labels)
    v_f1 = f1_score(all_val_output[:, 1], all_val_output[:, 0], average=None, zero_division=1, labels=labels)
        
    print("FINAL VALIDATION RESULTS: ")
    print("Accuracy: {:>4.2f} (%)".format(np.nanmean(v_acc) * 100))
    print("Precision: {:>4.2f} (%)".format(np.nanmean(v_prec) * 100))
    print("Recall: {:>4.2f} (%)".format(np.nanmean(v_rec) * 100))
    print("F1: {:>4.2f} (%)".format(np.nanmean(v_f1) * 100))
    
    print("FINAL VALIDATION RESULTS (PER CLASS): ")
    print("Accuracy: {0}".format(v_acc))
    print("Precision: {0}".format(v_prec))
    print("Recall: {0}".format(v_rec))
    print("F1: {0}".format(v_f1))

    print("GENERALIZATION GAP ANALYSIS: ")
    print("Train-Val-Accuracy Difference: {0}".format(np.nanmean(t_acc) - np.nanmean(v_acc)))
    print("Train-Val-Precision Difference: {0}".format(np.nanmean(t_prec) - np.nanmean(v_prec)))
    print("Train-Val-Recall Difference: {0}".format(np.nanmean(t_rec) - np.nanmean(v_rec)))
    print("Train-Val-F1 Difference: {0}".format(np.nanmean(t_f1) - np.nanmean(v_f1)))

    # save final postprocessed confusion matrix
    _, ax = plt.subplots(figsize=(15, 15), layout="constrained")
    ax.set_title('Confusion Matrix Total')
    conf_disp = ConfusionMatrixDisplay(confusion_matrix=v_conf_mat, display_labels=args.class_names)    
    conf_disp.plot(ax=ax, xticks_rotation='vertical', colorbar=False)
    mkdir_if_missing(os.path.join(log_dir, 'conf_mats'))
    plt.savefig(os.path.join(log_dir, 'conf_mats', 'all.png'))
    if run is not None:
        run['conf_matrices'].append(File(os.path.join(log_dir,  'conf_mats', 'all.png')), name='all')
    
    # submit final values to neptune 
    if run is not None:
        run['final_accuracy'] = np.nanmean(v_acc)
        run['final_precision'] = np.nanmean(v_prec)
        run['final_recall'] = np.nanmean(v_rec)
        run['final_f1'] = np.nanmean(v_f1)
        run['train-val-acc-diff'] = np.nanmean(t_acc) - np.nanmean(v_acc)
        run['train-val-prec-diff'] = np.nanmean(t_prec) - np.nanmean(v_prec)
        run['train-val-rec-diff'] = np.nanmean(t_rec) - np.nanmean(v_rec)
        run['train-val-f1-diff'] = np.nanmean(t_f1) - np.nanmean(v_f1)
    
    return net


def train_valid_split(train_data, valid_data, args, log_dir=None, run=None):
    """
    Method to apply normal cross-validation, i.e. one set split into train, validation and testing data.

    :param train_data: numpy array
        Data used for training
    :param valid_data: numpy array
        Data used for validation
    :param args: dict
        Args object containing all relevant hyperparameters and settings
    :param log_dir: string
        Logging directory
    :return pytorch model
        Trained network
    """
    print('\nCALCULATING TRAIN-VALID-SPLIT SCORES.\n')
    # Sensor data is segmented using a sliding window mechanism
    X_train, y_train = apply_sliding_window(train_data[:, :-1], train_data[:, -1],
                                            sliding_window_size=args.sw_length,
                                            unit=args.sw_unit,
                                            sampling_rate=args.sampling_rate,
                                            sliding_window_overlap=args.sw_overlap,
                                            )

    X_val, y_val = apply_sliding_window(valid_data[:, :-1], valid_data[:, -1],
                                        sliding_window_size=args.sw_length,
                                        unit=args.sw_unit,
                                        sampling_rate=args.sampling_rate,
                                        sliding_window_overlap=args.sw_overlap,
                                        )

    X_train, X_val = X_train[:, :, 1:], X_val[:, :, 1:]

    args.window_size = X_train.shape[1]
    args.nb_channels = X_train.shape[2]

    # network initialization
    if args.network == 'deepconvlstm':
        net = DeepConvLSTM(config=vars(args))
    elif args.network == 'attendanddiscriminate':        
            net = AttendAndDiscriminate(args.nb_channels, args.nb_classes, args.nb_units_lstm, args.nb_filters, args.filter_width, args.nb_layers_lstm, False, args.drop_prob, 0.5, 0.5, 'ReLU', 1, args.gpu)
    else:
        print("Did not provide a valid network name!")

    # optimizer initialization
    opt = init_optimizer(net, args)

    # optimizer initialization
    loss = init_loss(args)

    # lr scheduler initialization
    if args.adj_lr:
        print('Adjusting learning rate according to scheduler: ' + args.lr_scheduler)
        scheduler = init_scheduler(opt, args)
    else:
        scheduler = None

    net, checkpoint, val_output, train_output = train(X_train, y_train, X_val, y_val,
                                                      network=net, optimizer=opt, loss=loss, lr_scheduler=scheduler,
                                                      config=vars(args), run=run, name='split'
                                                      )

    if args.save_checkpoints:
        print('Saving checkpoint...')
        if args.valid_epoch == 'last':
            c_name = os.path.join(log_dir, "checkpoint_last_{}.pth".format(str(args.name)))
        else:
            c_name = os.path.join(log_dir, "checkpoint_best_{}.pth".format(str(args.name)))
        torch.save(checkpoint, c_name)

    if args.save_predictions:
        print('Saving predictions...')
        if args.valid_epoch == 'last':
            p_name = os.path.join(log_dir, "predictions_last_{}.csv".format(str(args.name)))
        else:
            p_name = os.path.join(log_dir, "predictions_best_{}.csv".format(str(args.name)))
        pd.DataFrame(val_output).to_csv(p_name)

    # fill values for normal evaluation
    labels = list(range(0, args.nb_classes))
    t_conf_mat = confusion_matrix(train_output[:, 1], train_output[:, 0], normalize='true', labels=labels)
    t_acc = t_conf_mat.diagonal()/t_conf_mat.sum(axis=1)
    t_prec = precision_score(train_output[:, 1], train_output[:, 0], average=None, zero_division=1, labels=labels)
    t_rec = recall_score(train_output[:, 1], train_output[:, 0], average=None, zero_division=1, labels=labels)
    t_f1 = f1_score(train_output[:, 1], train_output[:, 0], average=None, zero_division=1, labels=labels)
        
    v_conf_mat = confusion_matrix(val_output[:, 1], val_output[:, 0], normalize='true', labels=labels)
    v_acc = v_conf_mat.diagonal()/v_conf_mat.sum(axis=1)
    v_prec = precision_score(val_output[:, 1], val_output[:, 0], average=None, zero_division=1, labels=labels)
    v_rec = recall_score(val_output[:, 1], val_output[:, 0], average=None, zero_division=1, labels=labels)
    v_f1 = f1_score(val_output[:, 1], val_output[:, 0], average=None, zero_division=1, labels=labels)        
    
    print('VALIDATION RESULTS (macro): ')
    print("Accuracy: {:>4.2f} (%)".format(np.nanmean(v_acc) * 100))
    print("Precision: {:>4.2f} (%)".format(np.nanmean(v_prec) * 100))
    print("Recall: {:>4.2f} (%)".format(np.nanmean(v_rec) * 100))
    print("F1: {:>4.2f} (%)".format(np.nanmean(v_f1) * 100))

    print("VALIDATION RESULTS (PER CLASS): ")
    print("Accuracy: {0}".format(v_acc))
    print("Precision: {0}".format(v_prec))
    print("Recall: {0}".format(v_rec))
    print("F1: {0}".format(v_f1))

    print("GENERALIZATION GAP ANALYSIS: ")
    print("Train-Val-Accuracy Difference: {0}".format(np.nanmean(t_acc) - np.nanmean(v_acc)))
    print("Train-Val-Precision Difference: {0}".format(np.nanmean(t_prec) - np.nanmean(v_prec)))
    print("Train-Val-Recall Difference: {0}".format(np.nanmean(t_rec) - np.nanmean(v_rec)))
    print("Train-Val-F1 Difference: {0}".format(np.nanmean(t_f1) - np.nanmean(v_f1)))

    if args.save_analysis:
        tv_results = pd.DataFrame([v_acc, v_prec, v_rec, v_f1], columns=args.class_names)
        tv_results.index = ['accuracy', 'precision', 'recall', 'f1']
        tv_gap = pd.DataFrame([t_acc - v_acc, t_prec - v_prec, t_rec - v_rec, t_f1 - v_f1],
                              columns=args.class_names)
        tv_gap.index = ['accuracy', 'precision', 'recall', 'f1']
        tv_results.to_csv(os.path.join(log_dir, 'split_scores_{}.csv'.format(args.name)))
        tv_gap.to_csv(os.path.join(log_dir, 'tv_gap_{}.csv'.format(args.name)))

        if run is not None:    
            run["analysis"].upload(os.path.join(log_dir, 'split_scores_{}.csv'.format(args.name)))
            run["analysis"].upload(os.path.join(log_dir, 'tv_gap_{}.csv'.format(args.name)))
    
    # save final postprocessed confusion matrix
    _, ax = plt.subplots(figsize=(15, 15), layout="constrained")
    ax.set_title('Confusion Matrix Total')
    conf_disp = ConfusionMatrixDisplay(confusion_matrix=v_conf_mat, display_labels=args.class_names)    
    conf_disp.plot(ax=ax, xticks_rotation='vertical', colorbar=False)
    mkdir_if_missing(os.path.join(log_dir, 'conf_mats'))
    plt.savefig(os.path.join(log_dir, 'conf_mats', 'all.png'))
    if run is not None:
        run['conf_matrices'].append(File(os.path.join(log_dir, 'conf_mats', 'all.png')), name='all')
    
    # submit final values to neptune 
    if run is not None:
        run['final_accuracy'] = np.nanmean(v_acc)
        run['final_precision'] = np.nanmean(v_prec)
        run['final_recall'] = np.nanmean(v_rec)
        run['final_f1'] = np.nanmean(v_f1)
        run['train-val-acc-diff'] = np.nanmean(t_acc) - np.nanmean(v_acc)
        run['train-val-prec-diff'] = np.nanmean(t_prec) - np.nanmean(v_prec)
        run['train-val-rec-diff'] = np.nanmean(t_rec) - np.nanmean(v_rec)
        run['train-val-f1-diff'] = np.nanmean(t_f1) - np.nanmean(v_f1)
        
    return net
