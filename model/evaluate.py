##################################################
# All functions related to evaluating training and testing results
##################################################
# Author: Marius Bock
# Email: marius.bock(at)uni-siegen.de
##################################################

import numpy as np


def evaluate_participant_scores(participant_scores, gen_gap_scores, input_cm, class_names, nb_subjects, subjects,
                                filepath, filename, args):
    """
    Function which prints evaluation metrics of each participant, overall average and saves confusion matrix

    :param participant_scores: numpy array
        Array containing all results
    :param gen_gap_scores:
        Array containing generalization gap results
    :param input_cm: confusion matrix
        Confusion matrix of overall results
    :param class_names: list of strings
        Class names
    :param nb_subjects: int
        Number of subjects in dataset
    :param filepath: str
        Directory where to save plots to
    :param filename: str
        Name of plot
    :param args: dict
        Overall settings dict
    """
    print('\nPREDICTION RESULTS')
    print('-------------------')
    print('Average results')
    avg_acc = np.mean(participant_scores[0, :, :])
    std_acc = np.std(participant_scores[0, :, :])
    avg_prc = np.mean(participant_scores[1, :, :])
    std_prc = np.std(participant_scores[1, :, :])
    avg_rcll = np.mean(participant_scores[2, :, :])
    std_rcll = np.std(participant_scores[2, :, :])
    avg_f1 = np.mean(participant_scores[3, :, :])
    std_f1 = np.std(participant_scores[3, :, :])
    print('Avg. Accuracy {:.4f} (±{:.4f}), '.format(avg_acc, std_acc),
          'Avg. Precision {:.4f} (±{:.4f}), '.format(avg_prc, std_prc),
          'Avg. Recall {:.4f} (±{:.4f}), '.format(avg_rcll, std_rcll),
          'Avg. F1-Score {:.4f} (±{:.4f})'.format(avg_f1, std_f1))
    if args.include_void:
        print('Average results (no void)')
        avg_acc = np.mean(participant_scores[0, 1:, :])
        std_acc = np.std(participant_scores[0, 1:, :])
        avg_prc = np.mean(participant_scores[1, 1:, :])
        std_prc = np.std(participant_scores[1, 1:, :])
        avg_rcll = np.mean(participant_scores[2, 1:, :])
        std_rcll = np.std(participant_scores[2, 1:, :])
        avg_f1 = np.mean(participant_scores[3, 1:, :])
        std_f1 = np.std(participant_scores[3, 1:, :])
        print('Avg. Accuracy {:.4f} (±{:.4f}), '.format(avg_acc, std_acc),
              'Avg. Precision {:.4f} (±{:.4f}), '.format(avg_prc, std_prc),
              'Avg. Recall {:.4f} (±{:.4f}), '.format(avg_rcll, std_rcll),
              'Avg. F1-Score {:.4f} (±{:.4f})'.format(avg_f1, std_f1))
    print('Average class results')
    for i, class_name in enumerate(class_names):
        avg_acc = np.mean(participant_scores[0, i, :])
        std_acc = np.std(participant_scores[0, i, :])
        avg_prc = np.mean(participant_scores[1, i, :])
        std_prc = np.std(participant_scores[1, i, :])
        avg_rcll = np.mean(participant_scores[2, i, :])
        std_rcll = np.std(participant_scores[2, i, :])
        avg_f1 = np.mean(participant_scores[3, i, :])
        std_f1 = np.std(participant_scores[3, i, :])
        print('Class {}: Avg. Accuracy {:.4f} (±{:.4f}), '.format(class_name, avg_acc, std_acc),
              'Avg. Precision {:.4f} (±{:.4f}), '.format(avg_prc, std_prc),
              'Avg. Recall {:.4f} (±{:.4f}), '.format(avg_rcll, std_rcll),
              'Avg. F1-Score {:.4f} (±{:.4f})'.format(avg_f1, std_f1))
    print('Subject-wise results')
    for subject in range(nb_subjects):
        print('Subject ', subjects[subject], ' results: ')
        for i, class_name in enumerate(class_names):
            acc = participant_scores[0, i, subject]
            prc = participant_scores[1, i, subject]
            rcll = participant_scores[2, i, subject]
            f1 = participant_scores[3, i, subject]
            print('Class {}: Accuracy {:.4f}, '.format(class_name, acc),
                  'Precision {:.4f}, '.format(prc),
                  'Recall {:.4f}, '.format(rcll),
                  'F1-Score {:.4f}'.format(f1))

    print('\nGENERALIZATION GAP ANALYSIS')
    print('-------------------')
    print('Average results')
    avg_acc = np.mean(gen_gap_scores[0, :])
    std_acc = np.std(gen_gap_scores[0, :])
    avg_prc = np.mean(gen_gap_scores[1, :])
    std_prc = np.std(gen_gap_scores[1, :])
    avg_rcll = np.mean(gen_gap_scores[2, :])
    std_rcll = np.std(gen_gap_scores[2, :])
    avg_f1 = np.mean(gen_gap_scores[3, :])
    std_f1 = np.std(gen_gap_scores[3, :])
    print('Avg. Accuracy {:.4f} (±{:.4f}), '.format(avg_acc, std_acc),
          'Avg. Precision {:.4f} (±{:.4f}), '.format(avg_prc, std_prc),
          'Avg. Recall {:.4f} (±{:.4f}), '.format(avg_rcll, std_rcll),
          'Avg. F1-Score {:.4f} (±{:.4f})'.format(avg_f1, std_f1))
    print('Subject-wise results')
    for subject in range(nb_subjects):
        print('Subject ', subjects[subject], ' results: ')
        acc = gen_gap_scores[0, subject]
        prc = gen_gap_scores[1, subject]
        rcll = gen_gap_scores[2, subject]
        f1 = gen_gap_scores[3, subject]
        print('Accuracy {:.4f}, '.format(acc),
              'Precision {:.4f}, '.format(prc),
              'Recall {:.4f}, '.format(rcll),
              'F1-Score {:.4f}'.format(f1))
