import glob
from pathlib import Path

import numpy
import pandas
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import hamming
from scipy.spatial.distance import jaccard

from paths import root_dir
from constants import *

path = root_dir() + 'px_summary_dataset/'
Path(path).mkdir(parents=True, exist_ok=True)

max_files = 4


def main():
    bugs_df = load_bug_vectors()
    summary_df = bug_internal_smell_distance(bugs_df)
    summary_df.to_csv(path + 'smell_distances_within_bugs.csv')

    summary_df = pandas.read_csv(path + 'smell_distances_within_bugs.csv')
    summary_df = summary_df[summary_df['files_count'] <= max_files]

    bug_ids = []
    for pred_csv in glob.glob(
        root_dir() + 'p_FINAL_Bench4BL/p_model_for_smell_classification_bootstrap/**/075_final__predictions.csv'):
        pred_df = pandas.read_csv(pred_csv)
        bug_ids.extend(pred_df['bug_id'])
    bug_ids = list(set(bug_ids))

    renamed_bug_ids = []
    for bug_id in bug_ids:
        id = bug_id.split('_')[0]
        orga = bug_id.split('_')[1].split('--')[0]
        version = bug_id.split('--')[1]
        renamed_bug_ids.append(orga + '--' + version + '--' + id)
    summary_df = summary_df[summary_df['bug_id'].isin(renamed_bug_ids)]
    summary_df.groupby(by='project').mean().to_csv(path + 'smell_distances_within_bugs_averaged_by_project.csv')
    summary_df.mean().to_csv(path + 'smell_distances_within_bugs_averaged_total.csv')


def load_bug_vectors():
    all_bugs_df = pandas.DataFrame()
    for bug_file_vectors in glob.glob(root_dir() + 'bug_smell_vectors' + '/*--bug_smell_group_files_vectors.csv'):
        df = pandas.read_csv(bug_file_vectors)
        df = df[df['files_missing_in_src'] == 0]
        df = df.fillna('0')
        df = df[df['pmd_error'] == 0]
        df = df[df['cloc_error'] == 0]
        project_and_version = bug_file_vectors.split('/')[-1].rstrip('--bug_smell_group_files_vectors.csv')
        df['bug_id'] = project_and_version + '--' + df['bug_id'].astype(str)
        for trgt in TARGET_GROUPS:
            if trgt not in df.columns:
                df[trgt] = 0
            df[trgt] = df[trgt].astype(float).apply(lambda x: numpy.where(x > 0, 1, 0))
        all_bugs_df = pandas.concat([all_bugs_df, df])
    return all_bugs_df


def bug_internal_smell_distance(df):
    summary = []
    for bug_id in df['bug_id'].value_counts().index:
        bug_df = df[df['bug_id'] == bug_id].copy()
        orga = bug_id.split('--')[0]
        project = bug_id.split('--')[1].split('_')[0]

        files_count = len(bug_df)

        file_vectors = bug_df[TARGET_GROUPS].to_numpy()
        jaccard_sum = 0
        hamming_sum = 0
        euclidean_sum = 0
        num_op = 0
        for i in range(0, len(file_vectors) - 1):
            for j in range(i+1, len(file_vectors)):
                jaccard_sum += jaccard(file_vectors[i], file_vectors[j])
                hamming_sum += hamming(file_vectors[i], file_vectors[j])
                euclidean_sum += euclidean(file_vectors[i], file_vectors[j])
                num_op += 1
        resd = {
                'bug_id': bug_id,
                'project': project,
                'orga': orga,
                'jaccard_avg_op': 0,
                'hamming_avg_op': 0,
                'euclidean_avg_op': 0,
                'jaccard_avg_file': 0,
                'hamming_avg_file': 0,
                'euclidean_avg_file': 0,
                'jaccard_sum': jaccard_sum,
                'hamming_sum': hamming_sum,
                'euclidean_sum': euclidean_sum,
                'files_count': files_count,
                'op_count': num_op
        }
        if num_op:
            resd.update({
                'jaccard_avg_op': jaccard_sum / num_op,
                'hamming_avg_op': hamming_sum / num_op,
                'euclidean_avg_op': euclidean_sum / num_op,
                'jaccard_avg_file': jaccard_sum / files_count,
                'hamming_avg_file': hamming_sum / files_count,
                'euclidean_avg_file': euclidean_sum / files_count})
        summary.append(resd)
    return pandas.DataFrame(summary)

if __name__ == '__main__':
    main()
