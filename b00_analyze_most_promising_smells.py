import glob
import math

import numpy
import pandas
from paths import BENCH4BL_DIR, root_dir
from pathlib import Path

from utils.Logger import Logger
from utils.bench4bl_utils import rename_java_file

path = root_dir() + 'h_analyze_most_promising_smells/'
log = Logger()


def main():
    collect_bug_results(None, group='group')

    eval_collected_results('ap_bug_scores_kNone_group.csv')


def versions_train_test_split(df, test_fraction):
    versions = df.groupby('version').first()
    versions['orga_project'] = versions['orga'] + '--' + versions['project']
    versions = versions.reset_index()

    train_versions = []
    test_versions = []
    for orga_project in versions['orga_project'].value_counts().index.to_list():
        ver_list = versions[versions['orga_project'] == orga_project]['version'].value_counts().sort_index().index.to_list()
        ver_split = math.ceil(len(ver_list) * test_fraction)
        train_versions.extend(ver_list[:ver_split])
        test_versions.extend(ver_list[ver_split:])

    train_df = df[df['version'].isin(train_versions)]
    test_df = df[df['version'].isin(test_versions)]

    return train_df, test_df


def eval_collected_results(csv):
    df = pandas.read_csv(path + csv)
    df, _ = versions_train_test_split(df, 0.5)
    # df = df[df['tool'] == tool]
    df = df.groupby(by='tool').mean().drop(columns='Unnamed: 0')
    # df.mean(axis=0)
    # .to_csv(path + tool + '_' + csv)
    for col in df.columns.drop('base'):
        df[col] = (df[col] - df['base'])/df['base'] *100

    df.to_csv(path + 'tool_summary_' + csv)
    df.loc[['BugLocator', 'BLIA', 'BRTracer']].mean().sort_values(ascending=False).to_csv(path + 'most_useful_from_top3_tools_' + csv)
    df.loc[['BugLocator', 'BLIA', 'BRTracer']].T.to_latex(path + 'most_useful_from_top3_tools_' + csv + '.tex')


def collect_bug_results(k=None, group=""):
    ap_results = []
    rr_results = []
    for tool in glob.glob(root_dir() + 'bench4bl_localization_results/*'):
        log.s(tool)
        for orga in glob.glob(tool + '/*'):
            log.s(orga)
            for project in glob.glob(orga + '/*'):
                for version in glob.glob(project + '/*'):
                    orga_str = version.split('/')[-3]
                    project_str = version.split('/')[-2]
                    version_str = version.split('/')[-1]
                    tool_str = version.split('/')[-4]
                    version_smells_df = get_version_smells(orga_str, version_str, project_str, group)
                    version_smells_df['filename'] = version_smells_df['filename'].apply(lambda x: rename_java_file(x))
                    ground_truth_files = pandas.read_csv(root_dir() + 'bench4bl_summary/' + orga_str + '--' + version_str + '--buggy_files.csv').rename(columns={'file': 'filename'})
                    for bug in glob.glob(version + '/*.csv'):
                        bug_id = int(bug.split('/')[-1].rstrip('.csv'))
                        bug_ground_truth = ground_truth_files[ground_truth_files['bug_id'] == bug_id]

                        # bug_ground_truth = bug_ground_truth[~(bug_ground_truth['mode'] == 'A')]
                        num_relevant = len(bug_ground_truth)
                        if num_relevant == 0:
                            continue

                        bug_df = pandas.read_csv(bug)

                        ground_truth_smells = bug_ground_truth.merge(version_smells_df, on='filename', how='left')
                        # biggest file fingerprint:
                        bug_fingerprint = ground_truth_smells.sort_values(by='cloc_normalization', ascending=False).reset_index(drop=True).drop(columns=['Unnamed: 0', 'version', 'bug_id', 'filename', 'mode', 'cloc_normalization']).loc[0].to_dict()

                        base_score = eval_bug(bug_df, bug_ground_truth, k, num_relevant)
                        ap = {'base': base_score['ap']}
                        rr = {'base': base_score['rr']}
                        for trgt in bug_fingerprint.keys():
                            smell_ranked = rank_files(bug_fingerprint, version_smells_df, mean_lse, [trgt])
                            reranked = rerank_files(bug_df, smell_ranked)
                            bug_score = eval_bug(reranked, bug_ground_truth, k, num_relevant)
                            ap.update({trgt: bug_score['ap']})
                            rr.update({trgt: bug_score['rr']})

                        ap.update({
                            'tool': tool_str,
                            'version': version_str,
                            'orga': orga_str,
                            'project': project_str,
                            'bug_id': bug_id})
                        rr.update({
                            'tool': tool_str,
                            'version': version_str,
                            'orga': orga_str,
                            'project': project_str,
                            'bug_id': bug_id})
                        ap_results.append(ap)
                        rr_results.append(rr)

    pandas.DataFrame(ap_results).to_csv(path + 'ap_bug_scores_k' + str(k) + '_' + group + '.csv')
    pandas.DataFrame(rr_results).to_csv(path + 'rr_bug_scores_k' + str(k) + '_' + group + '.csv')


def rank_files(bug_smells, version_files_df, loss, target_groups):
    files_df = version_files_df.copy()
    files_df = loss(files_df, bug_smells, target_groups)
    # files_df['filename'] = files_df['filename'].astype(str).apply(lambda x: rename_java_file(x))
    files_df = files_df[['filename', loss.__name__]]
    return files_df


def rerank_files(tool_rank_df, smell_score_df):
    orig_weight = 0.8
    rank_df = tool_rank_df.copy()
    rank_df = rank_df.merge(smell_score_df, on='filename', how='left')
    rank_df['score'] = (1 - orig_weight) * rank_df['mean_lse'] + orig_weight * rank_df['score']
    rank_df = rank_df.sort_values(by='score', ascending=False)
    rank_df = rank_df.reset_index(drop=True)
    rank_df['rank'] = rank_df.index
    rank_df = rank_df.set_index('rank')
    return rank_df


def mean_lse(files_df, bug_smells, target_groups):
    for trgt in target_groups:
        files_df[trgt] = (files_df[trgt] - bug_smells[trgt])**2
    files_df['mean_lse'] = 1 - (files_df[target_groups].sum(axis=1)/len(target_groups))
    files_df = files_df.sort_values(by='mean_lse', ascending=False)
    files_df = files_df.reset_index()
    files_df.index.set_names('rank')
    return files_df


def get_version_smells(orga_str, version_str, project_str, group):
    if group:
        smells_path = root_dir() + 'pmd_results/' + orga_str + '--' + project_str + '--sources--' + version_str + '--smells_group_count_vector--cloc_normalized.csv'
    else:
        smells_path = root_dir() + 'pmd_results/' + orga_str + '--' + project_str + '--sources--' + version_str + '--smells_count_vector--cloc_normalized.csv'
    drop_cols = ['pmd_error', 'applied_cloc_normalization']

    smells_df = pandas.read_csv(smells_path).drop(drop_cols, axis=1)
    # smells_df = pandas.read_csv(smells_path)
    # smells_df = smells_groups_df.merge(smells_df, on='filename', how='left')

    for trgt in smells_df.columns.drop(['filename', 'cloc_normalization']):
        smells_df[trgt] = smells_df[trgt].apply(lambda x: numpy.where(x > 0, 1, 0))

    return smells_df


def rank_files_by_group_smells(bug_df, version_smells, bug_ground_truth):
    bug_df = bug_df.copy()
    bug_ground_truth = bug_ground_truth.copy().rename(columns={'file': 'filename'})
    bug_ground_truth = bug_ground_truth.merge(version_smells, on='filename', how='left')
    smells = bug_ground_truth.sort_values(by='cloc_normalization', ascending=False).loc[0].to_dict()
    # todo make ranking


def eval_bug(bug_df, bug_ground_truth, cut_off, num_relevant):
    result = {'cut_off': cut_off}

    bug_df['rank'] = bug_df.index + 1 # as rank starts with 0
    bug_df['is_buggy'] = bug_df['filename'].isin(bug_ground_truth['filename']).astype(int)

    bug_df = bug_df[:cut_off]

    top_rank = bug_df[bug_df['is_buggy'] == 1]['rank'].min()

    # top_x
    top_x = [1, 5, 10]
    for x in top_x:
        result.update({'top_' + str(x): int(top_rank <= x)})

    # RR
    result.update({'rr': 1./top_rank})

    # AP
    # number_of_positive_instances_in_results = len(bug_df[bug_df['is_buggy'] == 1])
    # ap = bug_df[bug_df['is_buggy'] == 1]['rank'].apply(lambda x: len(bug_df[(bug_df['is_buggy'] == 1) & (bug_df['rank'] <= x)])/x).sum() / number_of_positive_instances_in_results
    ap = bug_df[bug_df['is_buggy'] == 1]['rank'].apply(lambda x: len(bug_df[(bug_df['is_buggy'] == 1) & (bug_df['rank'] <= x)]) / x).sum() / num_relevant

    result.update({'ap': ap})

    return result


if __name__ == '__main__':
    main()
