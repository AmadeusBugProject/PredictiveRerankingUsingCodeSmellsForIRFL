import glob
import math
from pathlib import Path

import pandas
from matplotlib import pyplot as plt

from paths import root_dir
from utils.stats_utils import evaluate_bootstrap, get_box, t_test_x_greater_y, t_test_x_differnt_y

prefix_path = ''
path = root_dir() + prefix_path + 'p_summary_bootstrap/'
Path(path).mkdir(parents=True, exist_ok=True)



small_projects = [ # projects with less than 10 bugs in evaluation set
'MATH',
'SWF',
'SWARM',
'DATAGRAPH',
'HIVE',
'WFLY',
'SHDP',
'SOCIAL',
'CODEC',
'CONFIGURATION',
'SECOAUTH',
'SOCIALTW',
'BATCHADM',
'IO',
'JBMETA',
'LDAP',
'SOCIALFB',
'SPR',
]


def main():
    basic_stats(prefix_path + 'p_ranking_test_proejct_wise_bootstrap', 'project')
    project_wise_stats(prefix_path + 'p_ranking_test_proejct_wise_bootstrap', 'project_wise')
    project_wise_stats(prefix_path + 'p_ranking_test_proejct_wise_bootstrap', 'project_wise_others', combine_projects=small_projects)


def basic_stats(boostrap_folder, label):
    perf = []
    for summary in glob.glob(root_dir() + boostrap_folder + '/**/bug_scores_k' + 'None' + '_summary.csv'):
        df = pandas.read_csv(summary)
        for indx in [0, 2, 4]:
            tool = df.loc[indx]['label']
            ap_ext = df.loc[indx + 1]['ap@' + 'None']
            ap_orig = df.loc[indx]['ap@' + 'None']
            perf.append({'tool': tool, 'ap_ext': ap_ext, 'ap_orig': ap_orig})


    perf_df = pandas.DataFrame(perf)
    perf_df.to_csv(path + 'all_scores_' + label + '.csv')

    tools = perf_df['tool'].value_counts().index.to_list()

    ttests = pandas.DataFrame()
    for tool in tools:
        ttests = pandas.concat([ttests, t_test_x_greater_y(perf_df[perf_df['tool'] == tool]['ap_ext'], perf_df[perf_df['tool'] == tool]['ap_orig'], tool + '_ext', tool)])
        ttests = pandas.concat([ttests, t_test_x_differnt_y(perf_df[perf_df['tool'] == tool]['ap_ext'], perf_df[perf_df['tool'] == tool]['ap_orig'], tool + '_ext', tool)])
    ttests.to_csv(path + 'all_scores_' + label + '_ttest.csv')

    bootstrap_df = pandas.DataFrame()
    boxes = []
    for tool in tools:
        orig_b = evaluate_bootstrap(perf_df[perf_df['tool'] == tool]['ap_orig'], tool + ' orig', alpha=0.9)
        ext_b = evaluate_bootstrap(perf_df[perf_df['tool'] == tool]['ap_ext'], tool + ' ext', alpha=0.9)
        bootstrap_df = pandas.concat([bootstrap_df, pandas.DataFrame([orig_b, ext_b])], axis=0, ignore_index=True)
        boxes.append(get_box(orig_b))
        boxes.append(get_box(ext_b))

    bootstrap_df.to_csv(path + 'bootstrap_' + label + '.csv')
    bootstrap_df.set_index('label', drop=True).to_latex(path + 'bootstrap_' + label + '.tex', float_format="{:0.3f}".format)

    fig, ax1 = plt.subplots(figsize=(8, 4))
    boxplot1 = ax1.bxp(boxes, showfliers=False, widths=0.4, patch_artist=True, boxprops=dict(facecolor='lightcyan'), medianprops=dict(color="black", linewidth=1.5))
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(path + 'all_scores_' + label + '.pdf')


def project_wise_stats(boostrap_folder, label, combine_projects=None):
    all_scores = pandas.DataFrame()
    for summary in glob.glob(root_dir() + boostrap_folder + '/**/rank_test_samples_df.csv'):
        df = pandas.read_csv(summary)
        df['project'] = df['version_id'].apply(lambda x: x.split('--')[1].split('_')[0])
        if combine_projects:
            df['project'] = df['project'].replace({x: 'OTHER' for x in combine_projects})
        df['label'] = df['label'].apply(lambda x: x.split('_')[0] + '_ext' if 'mean_lse' in x else x)
        tools = df['label'].value_counts().index.to_list()
        tools.sort()
        tool_scores = pandas.DataFrame()
        for tool in tools:
            perf = df[df['label'] == tool].groupby(by='project').mean()['ap@' + 'None']
            tool_scores[tool] = perf
        tool_scores['count'] = df[df['label'] == tool]['project'].value_counts()
        all_scores = pandas.concat([all_scores, tool_scores.reset_index()], axis=0)

    all_scores.groupby(by='project').mean().to_csv(path + 'project_scores_' + label + '.csv')

    means_df = all_scores.groupby(by='project').mean()
    col_order = []
    tools = means_df.columns.drop('count')
    for tool_idx in list(range(0, 6, 2)):
        tool = tools[tool_idx]
        tool_ext = tool + '_ext'
        means_df[tool + '_percent_change'] = (means_df[tool_ext] - means_df[tool])/means_df[tool] * 100
        col_order += [tool, tool_ext, tool + '_percent_change']
    col_order.append('count')
    means_df[col_order].to_latex(path + 'project_scores_' + label + '.tex', float_format="{:0.3f}".format)

    ttest_for_projects_scores(tools, all_scores, label, 'ROO')
    ttest_for_projects_scores(tools, all_scores, label, 'HBASE')
    ttest_for_projects_scores(tools, all_scores, label, 'CAMEL')


def ttest_for_projects_scores(tools, all_scores, label, project):
    ttests = pandas.DataFrame()
    for tool_idx in list(range(0, 6, 2)):
        tool = tools[tool_idx]
        ttests = pandas.concat([ttests, t_test_x_greater_y(all_scores[all_scores['project']==project][tool + '_ext'], all_scores[all_scores['project']==project][tool], tool + '_ext', tool)])
        ttests = pandas.concat([ttests, t_test_x_differnt_y(all_scores[all_scores['project']==project][tool + '_ext'], all_scores[all_scores['project']==project][tool], tool + '_ext', tool)])
        btstrp_diff = all_scores[all_scores['project']==project][tool + '_ext'] - all_scores[all_scores['project']==project][tool]
        btstrp_count_ext_better = btstrp_diff.apply(lambda x: math.ceil(x)).sum() / len(btstrp_diff)
        ttests = pandas.concat([ttests, pandas.DataFrame({'h0': ['abs count ' + tool + '_ext' + ' better than ' + tool], 'test': ['paired bootstrap'], 'stat': [btstrp_diff.apply(lambda x: math.ceil(x)).sum()], 'p': [btstrp_count_ext_better]})])

    ttests.to_csv(path + 'project_scores_' + label + '_' + project + '_ttest.csv')


if __name__ == '__main__':
    main()
