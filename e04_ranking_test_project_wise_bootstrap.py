import glob
from pathlib import Path

import numpy
import pandas

from constants import *
from paths import root_dir
from utils.Logger import Logger
from utils.scoring_utils import score_bug, get_file_squared_errors

log = Logger()

path = root_dir() + 'p_ranking_test_proejct_wise_bootstrap/'
Path(path).mkdir(parents=True, exist_ok=True)


def main():
    for bootstrap_sample_path in glob.glob(root_dir() + 'p_model_for_smell_classification_bootstrap/*'):
        k = bootstrap_sample_path.split('/')[-1]
        Path(path + k).mkdir(parents=True, exist_ok=True)

        ranking_test_df = pandas.read_csv(bootstrap_sample_path + '/ranknet_test_df.csv')
        predictions_test = pandas.read_csv(bootstrap_sample_path + '/075_final__predictions.csv')

        best_weights_df = pandas.read_csv(root_dir() + 'p_ranking_training_bootstrap/' + k + '/best_params_per_project.csv')
        weights_d = best_weights_df[best_weights_df['metric'] == 'ap@' + 'None'][['tool', 'project', 'orig_weight']]


        rank_test_samples_df = ranking(ranking_test_df, predictions_test, weights_d)
        rank_test_samples_df.to_csv(path + k + '/rank_test_samples_df.csv')

        rank_test_samples_df = pandas.read_csv(path + k + '/rank_test_samples_df.csv')

        rank_test_samples_df = rank_test_samples_df.fillna(0)

        summary = rank_test_samples_df[['top_1', 'top_5', 'top_10', 'rr@10', 'ap@10', 'rr@20', 'ap@20', 'rr@None', 'ap@None', 'label']].groupby(by='label').mean()
        summary.to_csv(path + k + '/bug_scores_k' + 'None' + '_summary.csv')


def ranking(ranking_training_df, predictions_train, weights_d):
    predicitions = predictions_train[predictions_train['bug_id'].isin(ranking_training_df['bug_id'])]

    pred_df = predicitions.copy()
    pred_df['version_id'] = pred_df['bug_id'].apply(lambda x: '_'.join(x.split('_')[1:]))

    score_results_df = pandas.DataFrame()

    pmd_paths = list(glob.glob(root_dir() + 'pmd_results/*--smells_group_count_vector--cloc_normalized.csv'))
    for i, version_files_csv in enumerate(pmd_paths):
        version_id = version_files_csv.split('/')[-1].split('--')[0] + '--' + version_files_csv.split('/')[-1].split('--')[3]
        pred_versions = pred_df['version_id'].value_counts().index.to_list()
        if version_id in pred_versions:
            log.s(version_id + '\t\t' + str(i+1) + ' / ' + str(len(pmd_paths)))
            version_df = pred_df[pred_df['version_id'] == version_id]

            pmd_files_df = pandas.read_csv(version_files_csv).fillna(0)
            for trgt in TARGET_GROUPS:
                if trgt not in pmd_files_df.columns:
                    pmd_files_df[trgt] = 0
                else:
                    pmd_files_df[trgt] = pmd_files_df[trgt].apply(lambda x: numpy.where(x > 0, 1, 0))

            ground_truth_files_df = pandas.read_csv(root_dir() + 'bench4bl_summary/' + version_id + '--buggy_files.csv')
            for index, row in version_df.iterrows():
                for tool in TOOLS:
                    bug_score_results = rank_and_score_bug(version_id, row.to_dict(), pmd_files_df, ground_truth_files_df, tool, weights_d)
                    score_results_df = pandas.concat([score_results_df, bug_score_results])

    return score_results_df.reset_index(drop=True)


def rank_and_score_bug(version_id, bug_row, pmd_files_df, ground_truth_files_df, tool, weights_d):
    tool_path = root_dir() + 'bench4bl_localization_results/' + tool + '/'
    orga = version_id.split('--')[0]
    version = version_id.split('--')[1]
    project = version_id.split('--')[1].split('_')[0]
    bug_no = bug_row['bug_id'].split('_')[0]
    subdirs = orga + '/' + project + '/' + version + '/'

    tool_rank_path = tool_path + '/' + subdirs + bug_no + '.csv'
    if not Path(tool_rank_path).exists():
        return pandas.DataFrame()

    bug_ground_truth_files_df = ground_truth_files_df[ground_truth_files_df['bug_id'] == int(bug_no)].copy()
    bug_ground_truth_files_df = bug_ground_truth_files_df.rename(columns={'file': 'filename'})
    bug_ground_truth_files_df['relevant'] = 1

    # bug_ground_truth_files_df = bug_ground_truth_files_df[~(bug_ground_truth_files_df['mode'] == 'A')]
    num_relevant = len(bug_ground_truth_files_df)

    tool_rank_df = pandas.read_csv(tool_path + '/' + subdirs + bug_no + '.csv')
    tool_rank_df = tool_rank_df.drop(columns=['rank'])

    file_squaread_errors_df = get_file_squared_errors(bug_row, pmd_files_df)

    ranked_df = tool_rank_df.merge(file_squaread_errors_df, on='filename', how='left')
    ranked_df = ranked_df.merge(bug_ground_truth_files_df[['filename', 'relevant']], on='filename', how='left').fillna(0)

    ranked_df = ranked_df.sort_values(by='score', ascending=False).reset_index(drop=True)
    base_ranking = score_bug(ranked_df.copy(), num_relevant)
    base_ranking.update({'label': tool, 'tool': tool, 'bug_id': bug_row['bug_id'], 'version_id': bug_row['version_id']})

    rankings = [base_ranking]

    if len(weights_d[(weights_d['tool'] == tool) & (weights_d['project'] == orga + '--' + project)]):
        orig_weight = weights_d[(weights_d['tool'] == tool) & (weights_d['project'] == orga + '--' + project)]['orig_weight'].values[0]
    else:
        orig_weight = 1

    rank_df = ranked_df.copy()

    rank_df['score'] = (1 - orig_weight) * rank_df['mean_lse'] + orig_weight * rank_df['score']
    rank_df = rank_df.sort_values(by='score', ascending=False)
    rank_df = rank_df.reset_index(drop=True)
    new_ranking = score_bug(rank_df.copy(), num_relevant)
    new_ranking.update({'label': tool + '_' + 'mean_lse', 'tool': tool, 'bug_id': bug_row['bug_id'], 'version_id': bug_row['version_id']})
    rankings.append(new_ranking)

    return pandas.DataFrame(rankings)


if __name__ == '__main__':
    main()
