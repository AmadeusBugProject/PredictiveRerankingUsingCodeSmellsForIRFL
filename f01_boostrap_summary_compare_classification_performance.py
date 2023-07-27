import glob
from pathlib import Path

import numpy
import pandas
from sklearn import metrics

from paths import root_dir
from constants import *

prefix_path = ''
path = root_dir() + prefix_path + 'p_summary_bootstrap/'
Path(path).mkdir(parents=True, exist_ok=True)


def main():
    classification_stats()


def classification_stats():
    overall_performance_scores = pandas.DataFrame()
    project_performance_scores = pandas.DataFrame()
    for pred_path in glob.glob(root_dir() + prefix_path + 'p_model_for_smell_classification_bootstrap/**/05_train__predictions.csv'):
        pred_df = pandas.read_csv(pred_path)
        target_df = pandas.read_csv(pred_path.replace('05_train__predictions.csv', 'ranknet_train_df.csv'))

        df = pred_df.join(target_df, how='inner', lsuffix='_pred', rsuffix='_target')
        score = eval_pred(
            df[[x + '_target' for x in TARGET_GROUPS]].to_numpy(),
            df[[x + '_pred' for x in TARGET_GROUPS]].to_numpy(),
            TARGET_GROUPS
        )
        overall_performance_scores = pandas.concat([overall_performance_scores, pandas.DataFrame(score)], ignore_index=True)

        df['project'] = df['version'].apply(lambda x: x.split('--')[1].split('_')[0])
        for project in df['project'].value_counts().index.to_list():
            score = eval_pred(
                df[df['project'] == project][[x + '_target' for x in TARGET_GROUPS]].to_numpy(),
                df[df['project'] == project][[x + '_pred' for x in TARGET_GROUPS]].to_numpy(),
                TARGET_GROUPS
            )
            score = pandas.DataFrame(score)
            score['project'] = project
            project_performance_scores = pandas.concat([project_performance_scores, score], ignore_index=True)

    overall_performance_scores.to_csv(path + 'overall_classifier_performance_final_test_set.csv')
    overall_performance_scores.groupby(by='metric').mean().to_csv(path + 'mean_overall_classifier_performance_final_test_set.csv')
    project_performance_scores.to_csv(path + 'project_wise_classifier_performance_final_test_set.csv')
    project_performance_scores.groupby(by=['project', 'metric']).mean().to_csv(path + 'mean_project_wise_classifier_performance_final_test_set.csv')


def eval_pred(y_test, y_pred, target_cols):
    y_pred = numpy.where(y_pred < 0.5, 0, 1)
    f1_record = {'metric': 'f1'}
    roc_auc_record = {'metric': 'rocauc'}
    precision_record = {'metric': 'precision'}
    recall_record = {'metric': 'recall'}

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

    return [f1_record, roc_auc_record, precision_record, recall_record, test_count_record, test_fraction_record]


if __name__ == '__main__':
    main()
