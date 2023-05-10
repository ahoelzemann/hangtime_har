##################################################
# All functions related to preprocessing and loading data
##################################################
# Author: Marius Bock
# Email: marius.bock(at)uni-siegen.de
# Author: Alexander Hölzemann
# Email: alexander.hoelzemann(at)uni-siegen.de
##################################################

import os

import pandas as pd
import numpy as np
from sklearn import preprocessing

pd.options.mode.chained_assignment = None


def load_dataset(test_type, test_case, include_void=False):
    """
    Main function to load one of the supported datasets
    :param test_type: string

    :param test_case: string

    :param include_void: boolean, default: False
        Whether to include void class in dataframe
    :return: numpy float arrays, int, list of strings, int, boolean
        features, labels, number of classes, class names, sampling rate and boolean has_void
    """
    sampling_rate = 50
    has_void = True
    class_names = ['dribbling', 'shot', 'pass', 'rebound', 'layup', 'walking', 'running', 'standing', 'sitting']

    data_drill = pd.read_csv(os.path.join('data/', 'hangtime_drill_data.csv'), sep=',', header=None, index_col=None)
    data_warmup = pd.read_csv(os.path.join('data/', 'hangtime_warmup_data.csv'), sep=',', header=None, index_col=None)
    data_game = pd.read_csv(os.path.join('data/', 'hangtime_game_data.csv'), sep=',', header=None, index_col=None)
    data = pd.concat((data_drill, data_warmup, data_game), axis=0)
    data_dandw = pd.concat((data_drill, data_warmup), axis=0)

    subjects = data.iloc[:, 3].unique()
    le = preprocessing.LabelEncoder()
    le.fit(subjects)
    data.iloc[:, 3] = le.transform(data.iloc[:, 3])
    data_dandw.iloc[:, 3] = le.transform(data_dandw.iloc[:, 3])
    data_drill.iloc[:, 3] = le.transform(data_drill.iloc[:, 3])
    data_game.iloc[:, 3] = le.transform(data_game.iloc[:, 3])

    if test_type == 'hypertuning':
        data = data.iloc[:, 3:]
        X, y = preprocess_data(data, has_void, include_void)
        train = np.concatenate((X, np.expand_dims(y, axis=1)), axis=1)
        valid = None
    elif test_type == 'location_specific':
        data_us = data[data.iloc[:, 0] == 'us'].iloc[:, 3:]
        data_eu = data[data.iloc[:, 0] == 'eu'].iloc[:, 3:]
        X_us, y_us = preprocess_data(data_us, has_void, include_void)
        X_eu, y_eu = preprocess_data(data_eu, has_void, include_void)
        if test_case == 'split_USvsEU':
            train = np.concatenate((X_us, np.expand_dims(y_us, axis=1).astype(np.uint8)), axis=1)
            valid = np.concatenate((X_eu, np.expand_dims(y_eu, axis=1).astype(np.uint8)), axis=1)
        elif test_case == 'split_EUvsUS':
            train = np.concatenate((X_eu, np.expand_dims(y_eu, axis=1).astype(np.uint8)), axis=1)
            valid = np.concatenate((X_us, np.expand_dims(y_us, axis=1).astype(np.uint8)), axis=1)
    elif test_type == 'skill_specific':
        data_novice = data[data.iloc[:, 1] == 'novice'].iloc[:, 3:]
        data_expert = data[data.iloc[:, 1] == 'expert'].iloc[:, 3:]
        data_expert_eu = data[(data.iloc[:, 1] == 'expert') & (data.iloc[:, 0] == 'eu')].iloc[:, 3:]
        X_novice, y_novice = preprocess_data(data_novice, has_void, include_void)
        X_expert, y_expert = preprocess_data(data_expert, has_void, include_void)
        X_expert_eu, y_expert_eu = preprocess_data(data_expert_eu, has_void, include_void)

        if test_case == 'split_ExvsNo':
            train = np.concatenate((X_expert, np.expand_dims(y_expert, axis=1).astype(np.uint8)), axis=1)
            valid = np.concatenate((X_novice, np.expand_dims(y_novice, axis=1).astype(np.uint8)), axis=1)
        elif test_case == 'loso_Experts':
            train = np.concatenate((X_expert, np.expand_dims(y_expert, axis=1).astype(np.uint8)), axis=1)
            valid = None
        elif test_case == 'loso_ExEU':
            train = np.concatenate((X_expert_eu, np.expand_dims(y_expert_eu, axis=1).astype(np.uint8)), axis=1)
            valid = None
        elif test_case == 'split_NovsEx':
            train = np.concatenate((X_novice, np.expand_dims(y_novice, axis=1).astype(np.uint8)), axis=1)
            valid = np.concatenate((X_expert, np.expand_dims(y_expert, axis=1).astype(np.uint8)), axis=1)
    elif test_type == 'subset_specific':
        data_us = data[data.iloc[:, 0] == 'us'].iloc[:, 3:]
        data_eu = data[data.iloc[:, 0] == 'eu'].iloc[:, 3:]
        data_dandw = data_dandw.iloc[:, 3:]
        data_game = data_game.iloc[:, 3:]
        X_g, y_g = preprocess_data(data_game, has_void, include_void)
        X_us, y_us = preprocess_data(data_us, has_void, include_void)
        X_eu, y_eu = preprocess_data(data_eu, has_void, include_void)
        X_dandw, y_dandw = preprocess_data(data_dandw, has_void, include_void)
        if test_case == 'loso_US':
            train = np.concatenate((X_us, np.expand_dims(y_us, axis=1).astype(np.uint8)), axis=1)
            valid = None
        elif test_case == 'loso_EU':
            train = np.concatenate((X_eu, np.expand_dims(y_eu, axis=1).astype(np.uint8)), axis=1)
            valid = None
        elif test_case == 'loso_DandW':
            train = np.concatenate((X_dandw, np.expand_dims(y_dandw, axis=1).astype(np.uint8)), axis=1)
            valid = None
        elif test_case == 'loso_G':
            train = np.concatenate((X_g, np.expand_dims(y_g, axis=1).astype(np.uint8)), axis=1)
            valid = None
    elif test_type == 'session_specific':
        data_dandw = data_dandw.iloc[:, 3:]
        data_drill = data_drill.iloc[:, 3:]
        data_game = data_game.iloc[:, 3:]
        X_dandw, y_dandw = preprocess_data(data_dandw, has_void, include_void)
        X_d, y_d = preprocess_data(data_drill, has_void, include_void)
        X_g, y_g = preprocess_data(data_game, has_void, include_void)
        if test_case == 'split_DvsG':
            train = np.concatenate((X_d, np.expand_dims(y_d, axis=1).astype(np.uint8)), axis=1)
        elif test_case == 'split_DandWvsG':
            train = np.concatenate((X_dandw, np.expand_dims(y_dandw, axis=1).astype(np.uint8)), axis=1)
        valid = np.concatenate((X_g, np.expand_dims(y_g, axis=1).astype(np.uint8)), axis=1)

    if has_void and include_void:
        class_names = ['void'] + class_names

    return train, valid, subjects, len(class_names), class_names, sampling_rate, has_void


def preprocess_data(data, has_void=False, include_void=True):
    """
    Function to preprocess the wetlab dataset according to settings.
    :param data: pandas dataframe
        Dataframe containing all data
    :param has_void: boolean, default: False
        Boolean signaling whether dataset has a void class
    :param include_void: boolean, default: True
        Boolean signaling whether to include or not include the void class in the dataset
    :return numpy float arrays
        Training and validation datasets that can be used for training
    """
    print('Processing dataset files ...')
    if has_void:
        if include_void:
            pass
        else:
            data = data[(data.iloc[:, -1] != 'void_class')]

    X, y = data.iloc[:, :-1], adjust_labels(data.iloc[:, -1]).astype(int)

    # if no void class in dataset subtract one from all labels
    if has_void and not include_void:
        y -= 1

    print("Full dataset with size: | X {0} | y {1} | ".format(X.shape, y.shape))

    return X.astype(np.float32), y.astype(np.uint8)


def adjust_labels(data_y):
    """
    Transforms original labels into the range [0, nb_labels-1]

    :param data_y: numpy integer array
        Sensor labels
    :return: numpy integer array
        Modified sensor labels
    """
    data_y[data_y == "void"] = 0
    data_y[data_y == 'dribbling'] = 1
    data_y[data_y == 'shot'] = 2
    data_y[data_y == 'pass'] = 3
    data_y[data_y == 'rebound'] = 4
    data_y[data_y == 'layup'] = 5
    data_y[data_y == 'walking'] = 6
    data_y[data_y == 'running'] = 7
    data_y[data_y == 'standing'] = 8
    data_y[data_y == 'sitting'] = 9

    return data_y


def get_last_non_nan(series, index, missingvalues=1):
    """
    Get value of last non-NaN value in a series

    :param series: pandas series
        Input data
    :param index: int
        Index from where to start checking
    :param missingvalues: int, default: 1
        Number of missing values
    :return:
        Value of last non-NaN in a series
    """
    if not pd.isna(series[index - 1]):
        return series[index - 1], missingvalues
    else:
        return get_last_non_nan(series, index - 1)


def get_next_non_nan(series, index, missingvalues=1):
    """
    Get value of next non-NaN value in a series

    :param series: pandas series
        Input data
    :param index: int
        Index from where to start checking
    :param missingvalues: int, default: 1
        Number of missing values
    :return: series value, int
        Value of next non-NaN in a series
    """
    if not pd.isna(series[index + 1]):
        return series[index + 1], missingvalues
    else:
        return get_next_non_nan(series, index + 1, missingvalues=missingvalues + 1)


def replace_nan_values(series, output_dtype='float'):
    """
    Function to replace NaN values in a series

    :param series: data series
        Data to be filled
    :param output_dtype: string, default: float
        Output datatype of series
    :return: pandas series
        Filled series (transposed)
    """
    if output_dtype == 'float':
        if series is not np.array:
            series = np.array(series)
        if pd.isna(series[0]):
            series[0] = series[1]
        if pd.isna(series[series.shape[0] - 1]):
            lastNonNan, numberOfMissingValues = get_last_non_nan(series, series.shape[0] - 1)
            if numberOfMissingValues != 1:
                for k in range(1, numberOfMissingValues):
                    series[series.shape[0] - 1 - k] = lastNonNan
            series[series.shape[0] - 1] = series[series.shape[0] - 2]
        for x in range(0, series.shape[0]):
            if pd.isna(series[x]):
                lastNonNan, _ = get_last_non_nan(series, x)
                nextNonNan, _ = get_next_non_nan(series, x)
                missingValue = (lastNonNan + nextNonNan) / 2
                series[x] = missingValue

    elif output_dtype == 'int' or output_dtype == 'string':
        if series is not np.array:
            series = np.array(series)
        if pd.isna(series[0]):
            series[0] = series[1]
        if pd.isna(series[series.shape[0] - 1]):
            lastNonNan, numberOfMissingValues = get_last_non_nan(series, series.shape[0] - 1)
            if numberOfMissingValues != 1:
                for k in range(1, numberOfMissingValues):
                    series[series.shape[0] - 1 - k] = lastNonNan
            series[series.shape[0] - 1] = series[series.shape[0] - 2]
        for x in range(0, series.shape[0]):
            if pd.isna(series[x]):
                lastNonNan, missingValuesLast = get_last_non_nan(series, x)
                nextNonNan, missingValuesNext = get_next_non_nan(series, x)
                if missingValuesLast < missingValuesNext:
                    series[x] = lastNonNan
                else:
                    series[x] = nextNonNan
    else:
        print("Please choose a valid output dtype. You can choose between float, int and string.")
        exit(0)

    return series.T


def interpolate(data, freq_old, freq_new):
    """
    Function which changes the sampling rate of some data via interpolation

    :param data: numpy array
        Data to be resampled
    :param freq_old: int
        Old sampling rate
    :param freq_new: int
        New sampling rate
    :return: numpy float array
        Resampled data
    """
    tsAligned = np.divide(np.arange(0, data.shape[0]), freq_old)
    timeStep = 1 / freq_new
    tsCount = round(tsAligned[-1] / timeStep)
    tsMax = tsCount * timeStep
    tsNew = np.linspace(tsAligned[0], tsMax, tsCount + 1)
    dataNew = np.interp(tsNew, tsAligned, data)

    return tsNew, dataNew
