##################################################
# All functions related to creating datasets
##################################################
# Author: Marius Bock
# Email: marius.bock(at)uni-siegen.de
##################################################
import ast
import json

import pandas as pd
from glob import glob
import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None


def read_meta(file_dir):
    file = open(file_dir, "r")
    data = json.load(file)
    file.close()

    return data


def create_hangtime_data(raw_dir, save_dir):
    filenames = sorted(list(glob(os.path.join(raw_dir, '*.csv'))))
    meta_data = read_meta(os.path.join(raw_dir, 'meta.txt'))

    output_data_drill, output_data_warmup, output_data_game = None, None, None
    all_data = None

    for filename in filenames:
        print("Processing {}...".format(filename))
        sbj_name = filename.split('/')[-1].split('.')[0]
        sbj_meta = meta_data[sbj_name]
        sbj_location = sbj_meta['location']
        sbj_level = sbj_meta['skill']
        sbj_gender = sbj_meta['gender']
        sbj_data = pd.read_csv(filename)
        sbj_data = sbj_data[(sbj_data['coarse'] != 'not_labeled')]

        sbj_data['skill'] = sbj_level
        sbj_data['gender'] = sbj_gender
        sbj_data['location'] = sbj_location

        sbj_data = sbj_data.drop(['timestamp', 'in/out'], axis=1).reset_index()
        sbj_data = sbj_data[['location', 'skill', 'gender', 'subject', 'acc_x', 'acc_y', 'acc_z', 'basketball', 'locomotion', 'coarse']]
        sbj_data['subject'] = sbj_name
        sbj_all = sbj_data
        
        print('LABEL DISTRIBUTION ({})'.format(sbj_name))
        print('\nBASKETBALL: \n')
        print(sbj_data['basketball'].value_counts())
        print('\nLOCOMOTION: \n')
        print(sbj_data['locomotion'].value_counts())


        for i, row in sbj_data.iterrows():
            if row['basketball'] == 'not_labeled':
                sbj_data.iloc[i, -3] = row['locomotion']

        sbj_data = sbj_data[(sbj_data['basketball'] != 'not_labeled') & (sbj_data['basketball'] != 'jumping')]
        sbj_data = sbj_data.drop(['locomotion'], axis=1)
        sbj_output_drill = sbj_data[(sbj_data['coarse'] != 'game') & (sbj_data['coarse'] != 'warmup')].drop(['coarse'], axis=1)
        sbj_output_warmup = sbj_data[(sbj_data['coarse'] == 'warmup')].drop(['coarse'], axis=1)
        sbj_output_game = sbj_data[(sbj_data['coarse'] == 'game')].drop(['coarse'], axis=1)

        if output_data_drill is None:
            output_data_drill = sbj_output_drill
            output_data_warmup = sbj_output_warmup
            output_data_game = sbj_output_game
            all_data = sbj_all
        else:
            output_data_drill = pd.concat((output_data_drill, sbj_output_drill))
            output_data_warmup = pd.concat((output_data_warmup, sbj_output_warmup))
            output_data_game = pd.concat((output_data_game, sbj_output_game))
            all_data = pd.concat((all_data, sbj_all))
            
    print('LABEL DISTRIBUTION (DATASET USED FOR TRAINING):')
    print('\nDRILL: \n')
    print(output_data_drill['basketball'].value_counts())
    print('\nWARMUP: \n')
    print(output_data_warmup['basketball'].value_counts())
    print('\nGAME: \n')
    print(output_data_game['basketball'].value_counts())
    
    print('LABEL DISTRIBUTION (RAW DATASET FILTERED FOR COARSE LABELING)')
    print('\nBASKETBALL: \n')
    print(all_data['basketball'].value_counts())
    print('\nLOCOMOTION: \n')
    print(all_data['locomotion'].value_counts())

    output_data_drill.to_csv(os.path.join(save_dir, 'hangtime_drill_data.csv'), header=False, index=False)
    output_data_warmup.to_csv(os.path.join(save_dir, 'hangtime_warmup_data.csv'), header=False, index=False)
    output_data_game.to_csv(os.path.join(save_dir, 'hangtime_game_data.csv'), header=False, index=False)


if __name__ == '__main__':
    raw_path = 'data/raw'
    save_path = 'data'

    # hangtime
    create_hangtime_data(raw_path, save_path)
