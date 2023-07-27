import glob
from pathlib import Path

import numpy
import pandas

from paths import root_dir
from constants import *

path = root_dir() + 'p_ranking_training_bootstrap/'
Path(path).mkdir(parents=True, exist_ok=True)


def main():
    for bootstrap_sample_path in glob.glob(root_dir() + 'p_ranking_training_bootstrap/*'):
        rank_train_df = pandas.read_csv(bootstrap_sample_path + '/ranking_performance_train_df.csv')

        rank_train_df['project'] = rank_train_df['version_id'].apply(lambda x: x.split('_')[0])
        projects = rank_train_df['project'].value_counts().index.to_list()

        best_params = []
        for k in [10, 20, None]:
            for tool in TOOLS:
                for project in projects:
                    metric = 'ap@' + str(k)
                    train_scores_df = rank_train_df[(rank_train_df['tool'] == tool) & (rank_train_df['project'] == project)]
                    if len(train_scores_df):
                        best_idx = train_scores_df[[metric, 'tool', 'label', 'project']].groupby(by='label').mean()[metric].idxmax()

                        orig_weight = train_scores_df[train_scores_df['label'] == best_idx].mean()['orig_weight']

                        orig_tool_performance = train_scores_df[train_scores_df['label'] == tool].mean()[metric]
                        weighted_performance = train_scores_df[train_scores_df['label'] == best_idx].mean()[metric]
                        best_params.append({'tool': tool, 'project': project, 'orig_weight': orig_weight, 'orig_performance': orig_tool_performance, 'weighted_performance': weighted_performance, 'metric': metric})
                    else:
                        best_params.append({'tool': tool, 'project': project, 'orig_weight': 1, 'orig_performance': numpy.nan, 'weighted_performance': numpy.nan, 'metric': metric})

        k = bootstrap_sample_path.split('/')[-1]
        Path(path + k).mkdir(parents=True, exist_ok=True)

        pandas.DataFrame(best_params).to_csv(path + k + '/best_params_per_project.csv')


if __name__ == '__main__':
    main()
