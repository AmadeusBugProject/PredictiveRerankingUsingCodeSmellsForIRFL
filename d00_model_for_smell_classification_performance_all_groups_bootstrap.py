import json
from pathlib import Path

import numpy
import pandas
from sklearn import metrics

from constants import *
from paths import root_dir
from utils.Logger import Logger
from utils.nn_classifier import nn_predictor

log = Logger()

prefix_path = ''
path = root_dir() + prefix_path + 'p_model_for_smell_classification_bootstrap/'
Path(path).mkdir(parents=True, exist_ok=True)


def main():
    global path
    orig_path = path

    for k in range(0, BOOTSTRAP_REPEATS):
        path = orig_path + str(k) + '/'
        Path(path).mkdir(parents=True, exist_ok=True)

        train_df = pandas.read_csv(path + 'train_df.csv')
        ranknet_train_df = pandas.read_csv(path + 'ranknet_train_df.csv')
        with open(path + 'features_and_targets.json', 'r') as fd:
            features_and_targets = json.load(fd)
        feature_cols = features_and_targets['feature_cols']
        target_cols = features_and_targets['target_cols']

        make_mode_and_eval(train_df, ranknet_train_df, target_cols, feature_cols)


def make_mode_and_eval(train_df, test_df, target_cols, feature_cols):
    X_train = train_df[feature_cols].to_numpy()
    X_test = test_df[feature_cols].to_numpy()

    y_train = train_df[target_cols].to_numpy()
    y_test = test_df[target_cols].to_numpy()

    filenames_test = test_df['filename'].to_list()
    bug_id_test = test_df['bug_id'].to_list()

    model = nn_predictor(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_bin = numpy.where(y_pred < 0.5, 0, 1)

    predictions_df = pandas.DataFrame(index=test_df.index)
    for col in range(0, y_pred.shape[1]):
        predictions_df[target_cols[col]] = y_pred[:, col]

    predictions_df['filename'] = filenames_test
    predictions_df['bug_id'] = bug_id_test

    score = eval_pred(y_test, y_pred_bin, y_train, target_cols)
    pandas.DataFrame(score).to_csv(path + 'all_groups_scores_and_model_params_for_classification_evaluation.csv')


def eval_pred(y_test, y_pred, y_train, target_cols):
    f1_record = {'metric': 'f1'}
    roc_auc_record = {'metric': 'rocauc'}
    precision_record = {'metric': 'precision'}
    recall_record = {'metric': 'recall'}

    train_count_record = {'metric': 'train_count'}
    train_fraction_record = {'metric': 'train_fraction'}
    test_count_record = {'metric': 'test_count'}
    test_fraction_record = {'metric': 'test_fraction'}

    for col in range(0, y_pred.shape[1]):
        f1_record.update({target_cols[col]: metrics.f1_score(y_test[:, col], y_pred[:, col])})
        try:
            roc_auc = metrics.roc_auc_score(y_test[:, col], y_pred[:, col])
        except ValueError:
            roc_auc = 0
        roc_auc_record.update({target_cols[col]: roc_auc})

        precision_record.update({target_cols[col]: metrics.precision_score(y_test[:, col], y_pred[:, col])})
        recall_record.update({target_cols[col]: metrics.recall_score(y_test[:, col], y_pred[:, col])})

        test_count_record.update({target_cols[col]: y_test[:, col].sum()})
        test_fraction_record.update({target_cols[col]: y_test[:, col].sum()/len(y_test)})

        train_count_record.update({target_cols[col]: y_train[:, col].sum()})
        train_fraction_record.update({target_cols[col]: y_train[:, col].sum()/len(y_train)})

    return [f1_record, roc_auc_record, precision_record, recall_record, train_count_record, train_fraction_record, test_count_record, test_fraction_record]


if __name__ == '__main__':
    main()
