import numpy

from constants import *
from utils.bench4bl_utils import rename_java_file


def score_bug(bug_df, num_relevant):
    result = {}

    bug_df['rank'] = bug_df.index + 1 # as rank starts with 1

    top_rank = bug_df[bug_df['relevant'] == 1]['rank'].min()

    # top_x
    top_x = [1, 5, 10]
    for x in top_x:
        result.update({'top_' + str(x): int(top_rank <= x)})

    for cut_off in [10, 20, None]:
        bug_k_df = bug_df[:cut_off].copy()
        top_rank = bug_k_df[bug_k_df['relevant'] == 1]['rank'].min()

        # RR
        if numpy.isnan(top_rank):
            rr = 0
        else:
            rr = 1./top_rank

        result.update({'rr@' + str(cut_off): rr})

        # AP
        number_of_positive_instances_in_results = len(bug_k_df[bug_k_df['relevant'] == 1])
        if number_of_positive_instances_in_results == 0:
            ap = 0
        else:
            ap = bug_k_df[bug_k_df['relevant'] == 1]['rank'].apply(lambda x: len(bug_k_df[(bug_k_df['relevant'] == 1) & (bug_k_df['rank'] <= x)])/x).sum() / num_relevant
        result.update({'ap@' + str(cut_off): ap})

    return result


def get_file_squared_errors(bug_smells, version_files_df):
    files_df = version_files_df.copy()
    for trgt in TARGET_GROUPS:
        files_df[trgt] = (files_df[trgt] - bug_smells[trgt])**2

    files_df['mean_lse'] = 1 - (files_df[TARGET_GROUPS].sum(axis=1)/len(TARGET_GROUPS))
    files_df['filename'] = files_df['filename'].astype(str).apply(lambda x: rename_java_file(x))
    files_df = files_df[['filename', 'mean_lse'] + TARGET_GROUPS]
    return files_df