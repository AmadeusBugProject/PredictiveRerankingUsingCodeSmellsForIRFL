import glob
import math

import pandas

from paths import root_dir


def load_embedding_dataset():
    max_files = 4

    all_bugs_df = pandas.DataFrame()
    for bug_reports_csv in glob.glob(root_dir() + 'stackoverflow_mpnet_embeddings' + '/*--' + 'doc' + '.csv'):
        version_id = bug_reports_csv.split('/')[-1].replace('--' + 'doc' + '.csv', '')
        orga, version = version_id.split('--')
        project = version.split('_')[0]
        bug_reports_df = pandas.read_csv(bug_reports_csv)
        bug_reports_df['bug_id'] = bug_reports_df['bug_id'].astype(str) + '_' + version_id

        # drop duplicates
        bug_reports_df = bug_reports_df[bug_reports_df['duplicates'].isna()]

        bug_reports_df = bug_reports_df[bug_reports_df['text'] != '']
        bug_reports_df = bug_reports_df.drop(['Unnamed: 0', 'duplicates', 'is_duplicated_by'], axis=1)
        bug_reports_df = bug_reports_df.drop(
            set(bug_reports_df.columns).intersection({'sentence_embeddings', 'embeddings'}), axis=1)
        bug_reports_df = bug_reports_df.set_index('bug_id')

        bug_smells_csv = root_dir() + 'bug_smell_vectors/' + version_id + '--bug_smell_' + 'group_' + 'files' + '_vectors.csv'
        bug_smells_df = pandas.read_csv(bug_smells_csv)
        bug_smells_df['bug_id'] = bug_smells_df['bug_id'].astype(str) + '_' + version_id

        # Bug smell is biggest file, prioritizing production code over test files
        bug_smells_df['file_count'] = bug_smells_df.groupby('bug_id')['bug_id'].transform('size')
        bug_smells_df['is_prod_code'] = ~bug_smells_df['filename'].str.lower().str.contains('test') # take prod code over test code
        bug_smells_df = bug_smells_df[bug_smells_df['files_missing_in_src'] == 0]
        bug_smells_df = bug_smells_df.sort_values(['bug_id', 'is_prod_code', 'cloc_normalization']).drop_duplicates(
            subset=['bug_id'], keep='last')

        bug_smells_df = bug_smells_df[bug_smells_df['pmd_error'] == 0]
        bug_smells_df = bug_smells_df[bug_smells_df['cloc_error'] == 0]

        bug_smells_df = bug_smells_df[bug_smells_df['file_count'] <= max_files]
        drop_columns = {'Unnamed: 0', 'pmd_error', 'cloc_error', 'files_missing_in_src',
                        'cloc_normalization', 'is_prod_code'}

        bug_smells_df = bug_smells_df.drop(set(bug_smells_df.columns).intersection(drop_columns), axis=1)
        bug_smells_df = bug_smells_df.set_index('bug_id')

        bug_features_df = bug_smells_df.join(bug_reports_df, how='inner')

        all_bugs_df = pandas.concat([all_bugs_df, bug_features_df], axis=0)

    return all_bugs_df


def versions_train_test_split(df, feature_cols, target_cols, test_fraction, seed, bootstrap_samples_frac):
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
        sample = df[df['version'] == version].sample(frac=bootstrap_samples_frac, replace=True, random_state=seed)
        train_df = pandas.concat([train_df, sample], axis=0, ignore_index=False)
    X_train = train_df[feature_cols].to_numpy()
    y_train = train_df[target_cols].to_numpy()


    test_df = pandas.DataFrame()
    for version in test_versions:
        sample = df[df['version'] == version].sample(frac=bootstrap_samples_frac, replace=True, random_state=seed)
        test_df = pandas.concat([test_df, sample], axis=0, ignore_index=False)
    X_test = test_df[feature_cols].to_numpy()
    y_test = test_df[target_cols].to_numpy()

    return X_train, y_train, X_test, y_test, train_df, test_df