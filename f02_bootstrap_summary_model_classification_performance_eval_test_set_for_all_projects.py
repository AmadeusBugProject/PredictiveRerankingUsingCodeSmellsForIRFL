import glob
from pathlib import Path

import numpy
import pandas
from sklearn import metrics

from f00_bootstrap_summary_compare_map_and_ttest import small_projects
from paths import root_dir
from constants import *

prefix_path = ''
path = root_dir() + prefix_path + 'p_summary_classification_test_set/'
Path(path).mkdir(parents=True, exist_ok=True)


def main():
    classification_stats_for_project(other='')
    classification_stats_for_project(other='other')


def classification_stats_for_project(other=''):
    scores = pandas.DataFrame()
    for pred_csv in glob.glob(root_dir() + prefix_path + 'p_model_for_smell_classification_bootstrap/**/075_final__predictions.csv'):
        pred_df = pandas.read_csv(pred_csv)
        act_df = pandas.read_csv(pred_csv.replace('075_final__predictions.csv', 'final_classifier_test_df.csv'))

        df = pred_df[TARGET_GROUPS].join(act_df[TARGET_GROUPS + ['project']], lsuffix='pred', rsuffix='act')

        for project in df['project'].value_counts().index:
            dfp = df[df['project'] == project].copy()

            y_test = dfp[[x + 'act' for x in TARGET_GROUPS]].to_numpy()
            y_pred = dfp[[x + 'pred' for x in TARGET_GROUPS]].to_numpy()
            y_pred_bin = numpy.where(y_pred < 0.5, 0, 1)

            eval_b = eval_pred(y_test, y_pred_bin, TARGET_GROUPS, project)
            edf = pandas.DataFrame(eval_b)

            if project in small_projects and other:
                edf['project'] = 'OTHER'
            else:
                edf['project'] = project
            scores = pandas.concat([scores, edf], ignore_index=True)
    ma_df = scores.fillna(0).groupby(['project', 'metric']).mean()
    ma_df.T.mean().unstack().to_csv(path + 'macro_average_classification_performances_per_project' + other + '.csv')
    ma_df.T.mean().unstack().to_latex(path + 'macro_average_classification_performances_per_project' + other + '.tex', float_format="{:0.3f}".format)



def eval_pred(y_test, y_pred, target_cols, project):
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
        test_fraction_record.update({target_cols[col]: y_test[:, col].sum() / len(y_test)})

    return [f1_record, roc_auc_record, precision_record, recall_record, train_count_record, train_fraction_record,
            test_count_record, test_fraction_record]




if __name__ == '__main__':
    main()
