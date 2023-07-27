import json
import re
from pathlib import Path

import numpy
import pandas

from constants import *
from paths import root_dir
from utils.Logger import Logger
from utils.dataset_utils import load_embedding_dataset, versions_train_test_split

log = Logger()

path = root_dir() + 'p_model_for_smell_classification_bootstrap/'
Path(path).mkdir(parents=True, exist_ok=True)

def setup_train_test_sets(df, target_cols, feature_cols, seed):
    dff = df.copy()
    with open(path + 'seed.json', 'w') as fd:
        json.dump(seed, fd)

    with open(path + 'features_and_targets.json', 'w') as fd:
        json.dump({'feature_cols': feature_cols, 'target_cols': target_cols.to_list()}, fd)

    _, _, _, _, train_df, test_df = versions_train_test_split(dff, feature_cols, TARGET_GROUPS, 0.5, seed, BOOTSTRAP_SAMPLE_FRACTION)
    _, _, _, _, ranknet_train_df, ranknet_test_df = versions_train_test_split(test_df, feature_cols, TARGET_GROUPS, 0.5, seed, BOOTSTRAP_SAMPLE_FRACTION)

    train_df.to_csv(path + 'train_df.csv')
    test_df.to_csv(path + 'test_df.csv')
    ranknet_train_df.to_csv(path + 'ranknet_train_df.csv')
    ranknet_test_df.to_csv(path + 'ranknet_test_df.csv')

    final_classifier_train_df = pandas.concat([train_df, ranknet_train_df])
    final_classifier_test_df = ranknet_test_df.copy()
    final_classifier_train_df.to_csv(path + 'final_classifier_train_df.csv')
    final_classifier_test_df.to_csv(path + 'final_classifier_test_df.csv')

    return train_df, test_df, final_classifier_train_df, final_classifier_test_df, feature_cols


def main():
    global path
    orig_path = path
    df, target_cols, feature_cols = load_dataset()

    df['project'] = df['version'].apply(lambda x: x.split('--')[1].split('_')[0])
    # df = df[df['project'] == 'ROO']
    # df = df[df['project'] == 'HBASE']
    # df = df[df['project'] == 'CAMEL']

    for k in range(0, BOOTSTRAP_REPEATS):
        path = orig_path + str(k) + '/'
        Path(path).mkdir(parents=True, exist_ok=True)
        train_df, test_df, final_classifier_train_df, final_classifier_test_df, feature_cols = setup_train_test_sets(df, target_cols, feature_cols, k)


def load_dataset():
    all_bugs_df = load_embedding_dataset()
    embedding_length = len([x for x in list(all_bugs_df.columns) if re.match('^emb_\d+$', x)])
    target_cols = all_bugs_df.columns.drop(['emb_' + str(x) for x in range(0, embedding_length)] + ['version', 'text', 'filename', 'file_count'])
    feature_cols = ['emb_' + str(x) for x in range(0, embedding_length)]
    all_bugs_df = all_bugs_df[~all_bugs_df['version'].isna()]
    df = all_bugs_df.fillna(0)
    # ONE HOT ENCODE TARGET:
    for trgt in target_cols:
        df[trgt] = df[trgt].apply(lambda x: numpy.where(x > 0, 1, 0))

    log.s('loaded dataset with len: ' + str(len(df)))
    return df, target_cols, feature_cols


if __name__ == '__main__':
    main()
