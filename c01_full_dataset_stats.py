import glob
import math
import re
from pathlib import Path

import numpy
import pandas
from matplotlib import pyplot as plt
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS

from constants import *
from paths import root_dir
from utils.dataset_utils import load_embedding_dataset

path = root_dir() + 'px_summary_dataset/'
Path(path).mkdir(parents=True, exist_ok=True)


def main():
    group_size_and_text_stats()
    cloc_latest_versions()
    dataset_size()


def dataset_size():
    all_bugs_df = pandas.DataFrame()
    for bug_reports_csv in glob.glob(root_dir() + 'stackoverflow_mpnet_embeddings' + '/*--' + 'doc' + '.csv'):
        version_id = bug_reports_csv.split('/')[-1].replace('--' + 'doc' + '.csv', '')
        orga, version = version_id.split('--')
        project = version.split('_')[0]
        bug_reports_df = pandas.read_csv(bug_reports_csv)
        bug_reports_df['bug_id'] = bug_reports_df['bug_id'].astype(str) + '_' + version_id
        bug_reports_df['project'] = project
        bug_reports_df['version'] = version
        bug_reports_df['orga'] = orga


        all_bugs_df = pandas.concat([all_bugs_df, bug_reports_df], axis=0)
    bench4bl_all_bugs_len = len(all_bugs_df) # 9278
    bench4bl_all_versions_len = len(all_bugs_df['version'].value_counts()) # 9278

    df, target_cols, feature_cols = load_dataset()
    cleaned_dataset_len = len(df)
    cleaned_dataset_versions_len = len(df['version'].value_counts())

    train_len, test_len, ranknet_train_len, ranknet_test_len, train_version_len, test_version_len, ranknet_train_version_len, ranknet_test_version_len = size_train_test_sets(df, target_cols, feature_cols, 0)

    pandas.DataFrame([
        {'Dataset': 'Bench4BL', 'No. bugs': bench4bl_all_bugs_len, 'No. versions': bench4bl_all_versions_len},
        {'Dataset': 'Bench4BL cleaned', 'No. bugs': cleaned_dataset_len, 'No. versions': cleaned_dataset_versions_len},
        {'Dataset': 'Classification training set', 'No. bugs': train_len, 'No. versions': train_version_len},
        {'Dataset': 'Ranking training set', 'No. bugs': ranknet_train_len, 'No. versions': ranknet_train_version_len},
        {'Dataset': 'Test set', 'No. bugs': ranknet_test_len, 'No. versions': ranknet_test_version_len},
    ]).to_latex(path + 'datasetSplits.tex')


def size_train_test_sets(df, target_cols, feature_cols, seed):
    dff = df.copy()
    _, _, _, _, train_df, test_df = versions_train_test_split(dff, feature_cols, TARGET_GROUPS, 0.5, seed)
    _, _, _, _, ranknet_train_df, ranknet_test_df = versions_train_test_split(test_df, feature_cols, TARGET_GROUPS,
                                                                                                                           0.5, seed)
    return len(train_df), len(test_df), len(ranknet_train_df), len(ranknet_test_df),\
           len(train_df['version'].value_counts()), len(test_df['version'].value_counts()), len(ranknet_train_df['version'].value_counts()), len(ranknet_test_df['version'].value_counts())


def versions_train_test_split(df, feature_cols, target_cols, test_fraction, seed):
    versions = pandas.DataFrame({'version': df['version'].value_counts().index})
    # versions = versions.sort_values(by='version')
    versions['orga'] = versions['version'].astype(str).apply(lambda x: x.split('--')[0])
    versions['project'] = versions['version'].astype(str).apply(lambda x: x.split('--')[1].split('_')[0])
    versions['orga_project'] = versions['orga'] + '--' + versions['project']

    train_versions = []
    test_versions = []
    for orga_project in versions['orga_project'].value_counts().index.to_list():
        ver_list = versions[versions['orga_project'] == orga_project]['version'].value_counts().sort_index().index.to_list()
        ver_split = math.ceil(len(ver_list) * test_fraction)
        train_versions.extend(ver_list[:ver_split])
        test_versions.extend(ver_list[ver_split:])

    train_df = pandas.DataFrame()
    for version in train_versions:
        sample = df[df['version'] == version] # .sample(frac=bootstrap_samples_frac, replace=True, random_state=seed)
        train_df = pandas.concat([train_df, sample], axis=0, ignore_index=False)
    X_train = train_df[feature_cols].to_numpy()
    y_train = train_df[target_cols].to_numpy()


    test_df = pandas.DataFrame()
    for version in test_versions:
        sample = df[df['version'] == version] # .sample(frac=bootstrap_samples_frac, replace=True, random_state=seed)
        test_df = pandas.concat([test_df, sample], axis=0, ignore_index=False)
    X_test = test_df[feature_cols].to_numpy()
    y_test = test_df[target_cols].to_numpy()

    return X_train, y_train, X_test, y_test, train_df, test_df


def cloc_latest_versions():
    stats = []
    for cloc_csv in glob.glob(root_dir() + 'cloc_results/*--java_loc_vector.csv'):
        df = pandas.read_csv(cloc_csv)
        df = df[df['language']=='Java']
        orga = cloc_csv.split('/')[-1].rstrip('--java_loc_vector.csv').split('--')[0]
        project = cloc_csv.split('/')[-1].rstrip('--java_loc_vector.csv').split('--')[1]
        version = cloc_csv.split('/')[-1].rstrip('--java_loc_vector.csv').split('--')[-1]
        java_file_count = len(df)
        mean_java_loc = df['code'].mean()
        mean_all_loc = (df['code'] + df['comment'] + df['blank']).mean()
        stats.append({'orga': orga, 'project': project, 'version': version, 'java_file_count': java_file_count, 'mean_java_loc': mean_java_loc, 'mean_all_loc': mean_all_loc})

    stats_df = pandas.DataFrame(stats)
    latest_versions = []
    for project in stats_df['project'].value_counts().index.to_list():
        latest_versions.append(stats_df[stats_df['project'] == project]['version'].sort_values().to_list()[-1])
    latest_df = stats_df[stats_df['version'].isin(latest_versions)]

    stats = []
    bench4bl_stats = {'project': 'BENCH4BL'}
    bench4bl_stats.update({'java_file_count': latest_df['java_file_count'].mean()})
    bench4bl_stats.update({'mean_java_loc': latest_df['mean_java_loc'].mean()})
    bench4bl_stats.update({'mean_all_loc': latest_df['mean_all_loc'].mean()})
    stats.append(bench4bl_stats)

    for project in ['ROO', 'HBASE', 'CAMEL']:
        project_stats = {'project': project}
        project_stats.update({'java_file_count': latest_df[latest_df['project']==project]['java_file_count'].mean()})
        project_stats.update({'mean_java_loc': latest_df[latest_df['project']==project]['mean_java_loc'].mean()})
        project_stats.update({'mean_all_loc': latest_df[latest_df['project']==project]['mean_all_loc'].mean()})
        stats.append(project_stats)

    stats_df = pandas.DataFrame(stats)
    stats_df.to_csv(path + 'cloc_stats.csv')
    stats_df.set_index('project').to_latex(path + 'cloc_stats.tex', float_format="{:0.2f}".format)


def group_size_and_text_stats():
    df, target_cols, feature_cols = load_dataset()
    df['text_len'] = df['text'].apply(lambda x: len(x))
    df['project'] = df['version'].apply(lambda x: x.split('--')[1].split('_')[0])

    stats = []

    bench4bl_stats = df[TARGET_GROUPS].sum().to_dict()
    bench4bl_stats.update({'project': 'BENCH4BL'})
    bench4bl_stats.update({'size': len(df)})
    bench4bl_stats.update({'mean_bug_report_len': df['text'].apply(lambda x: len(x)).mean()})
    bench4bl_stats.update({'mean_file_count': df['file_count'].mean()})
    stats.append(bench4bl_stats)

    for project in ['ROO', 'HBASE', 'CAMEL']:
        project_stats = df[df['project'] == project][TARGET_GROUPS].sum().to_dict()
        project_stats.update({'project': project})
        project_stats.update({'size': len(df[df['project'] == project])})
        project_stats.update({'mean_bug_report_len': df[df['project'] == project]['text'].apply(lambda x: len(x)).mean()})
        project_stats.update({'mean_file_count': df[df['project'] == project]['file_count'].mean()})
        stats.append(project_stats)

    col_order = ['project', 'size', 'Performance', 'jpinpoint-common-rules', 'Best Practices',
       'Multithreading', 'jpinpoint-concurrent-rules',
       'mean_bug_report_len', 'mean_file_count']

    stats_df = pandas.DataFrame(stats)

    stats_df.to_csv(path + 'absolute_target_count_and_length_stats.csv')
    stats_df[col_order].set_index('project').T.to_latex(path + 'absolute_target_count_and_length_stats.tex', float_format="{:0.2f}".format)

    for target in TARGET_GROUPS:
        stats_df[target] = stats_df[target]/stats_df['size']*100
    stats_df.to_csv(path + 'fractional_target_count_and_length_stats.csv')
    stats_df[col_order].set_index('project').T.to_latex(path + 'fractional_target_count_and_length_stats.tex', float_format="{:0.2f}".format)
    stats_df.set_index('project')[TARGET_GROUPS].plot.bar()
    # plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left', ncol=1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left', ncol=1)
    plt.tight_layout()
    plt.savefig(path + 'fractional_target_count.pdf')

    all_df = df.copy()
    all_df['project'] = 'BENCH4BL'
    text_lenght_box_df = pandas.concat([all_df, df[df['project'].isin(['ROO', 'HBASE', 'CAMEL'])]].copy())
    text_lenght_box_df[['project', 'text_len']].groupby(by='project').describe().to_latex(path + 'bug_report_lengths.tex', float_format="{:0.2f}".format)
    text_lenght_box_df.boxplot(column='text_len', by='project', showfliers=False, grid=False)
    plt.savefig(path + 'bug_report_lengths.pdf')

    word_histogram(df, path + 'word_histogram_bench4bl.pdf')
    word_histogram(df[df['project']=='ROO'].copy(), path + 'word_histogram_roo.pdf')
    word_histogram(df[df['project']=='HBASE'].copy(), path + 'word_histogram_hbase.pdf')
    word_histogram(df[df['project']=='CAMEL'].copy(), path + 'word_histogram_camel.pdf')


def word_histogram(df, fig_path):
    all_text = ' '.join(df['text'].to_list())
    for spchar in ['.', ',', '!', '?', '(', ')', '{', '}']:
        all_text = all_text.replace(spchar, ' ')
    all_words = pandas.DataFrame({'words': all_text.split()})
    all_words['words'] = all_words['words'].str.lower()

    word_frequency = pandas.DataFrame({'word': all_words['words'].value_counts().index, 'count': all_words['words'].value_counts().values})
    word_frequency = word_frequency[word_frequency['word'].str.match(r'^[A-Za-z]+$')]
    word_frequency = word_frequency[~word_frequency['word'].isin(ENGLISH_STOP_WORDS)]

    df['words'] = df['text'].str.lower().str.split()
    word_frequency['one_hot'] = word_frequency['word'].apply(
        lambda x: len(df[df['words'].apply(lambda y: x in y)]))

    ax = word_frequency.sort_values('one_hot').tail(20).plot.barh(x='word', y='one_hot', legend=None)
    ax.set_xlabel('word frequency')
    ax.set_ylabel('')
    ax.set_title('Title')
    plt.tight_layout()
    plt.savefig(fig_path)


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

    return df, target_cols, feature_cols


if __name__ == '__main__':
    main()
