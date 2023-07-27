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

path = root_dir() + 'p_model_for_smell_classification_bootstrap/'
Path(path).mkdir(parents=True, exist_ok=True)


def main():
    global path
    orig_path = path

    for k in range(0, BOOTSTRAP_REPEATS):
        path = orig_path + str(k) + '/'
        Path(path).mkdir(parents=True, exist_ok=True)

        train_df = pandas.read_csv(path + 'train_df.csv')
        test_df = pandas.read_csv(path + 'test_df.csv')

        final_classifier_train_df = pandas.read_csv(path + 'final_classifier_train_df.csv')
        final_classifier_test_df = pandas.read_csv(path + 'final_classifier_test_df.csv')

        with open(path + 'features_and_targets.json', 'r') as fd:
            features_and_targets = json.load(fd)
        feature_cols = features_and_targets['feature_cols']
        target_cols = features_and_targets['target_cols']

        make_mode_and_predictions(train_df, test_df, feature_cols, '05_train_')
        make_mode_and_predictions(final_classifier_train_df, final_classifier_test_df, feature_cols, '075_final_')


def make_mode_and_predictions(train_df, test_df, feature_cols, label):
    target_cols = TARGET_GROUPS
    X_train = train_df[feature_cols].to_numpy()
    X_test = test_df[feature_cols].to_numpy()

    y_train = train_df[target_cols].to_numpy()
    y_test = test_df[target_cols].to_numpy()

    filenames_test = test_df['filename'].to_list()
    bug_id_test = test_df['bug_id'].to_list()

    model = nn_predictor(X_train, y_train)

    model.save(path + label + '_model.tf')

    y_pred = model.predict(X_test)
    y_pred_bin = numpy.where(y_pred < 0.5, 0, 1)

    predictions_df = pandas.DataFrame(index=test_df.index)
    for col in range(0, y_pred.shape[1]):
        predictions_df[TARGET_GROUPS[col]] = y_pred[:, col]

    predictions_df['filename'] = filenames_test
    predictions_df['bug_id'] = bug_id_test

    predictions_df.to_csv(path + label + '_predictions.csv')

    score = eval_pred(y_test, y_pred_bin, TARGET_GROUPS)

    score.update({'DOC': DOC,
                  'EMBEDDING': EMBEDDING,
                  'PREDICTED_SMELLS': PREDICTED_SMELLS,
                  'BUG_SMELL_FEATURES': BUG_SMELL_FEATURES,
                  'HIDDEN_LAYERS': HIDDEN_LAYERS,
                  'LAYER_WIDTH': LAYER_WIDTH,
                  'DROPOUT': DROPOUT,
                  'LOSS_FUNCTION': LOSS_FUNCTION,
                  'BATCH_SIZE': BATCH_SIZE})
    with open(path + label + '_scores_and_model_params.json', 'w') as fd:
        json.dump(score, fd)


def eval_pred(y_test, y_pred, target_cols):
    score = {}
    macro_f1 = 0
    for col in range(0, y_pred.shape[1]):
        macro_f1 += metrics.f1_score(y_test[:, col], y_pred[:, col])
        score.update({target_cols[col]: metrics.f1_score(y_test[:, col], y_pred[:, col])})
    score.update({'macro_f1': macro_f1/y_pred.shape[1]})
    return score


if __name__ == '__main__':
    main()
