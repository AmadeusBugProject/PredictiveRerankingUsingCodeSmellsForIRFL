import glob

import numpy as np
import pandas

from paths import root_dir


def main():
    error_reports = []
    for buggy_files_csv in glob.glob(root_dir() + 'bench4bl_summary/*--buggy_files.csv'):
        version_id = buggy_files_csv.split('/')[-1].replace('--buggy_files.csv', '')
        print(version_id)
        orga, version = version_id.split('--')
        project = version.split('_')[0]

        smells_file = '--'.join([orga, project, 'sources', version]) + '--smells_count_vector--cloc_normalized.csv'
        smells_group_file = '--'.join([orga, project, 'sources', version]) + '--smells_group_count_vector--cloc_normalized.csv'

        buggy_files_df = pandas.read_csv(buggy_files_csv)
        buggy_files_df['filename'] = buggy_files_df['file'].apply(lambda x: x.replace('.', '/').replace('/java', '.java'))
        buggy_files_df = buggy_files_df.set_index('filename')
        buggy_files_df = buggy_files_df[['bug_id']]

        smells_df = pandas.read_csv(root_dir() + 'pmd_results/' + smells_file)
        bug_smells_files, error_report = map_by_filename(smells_df, buggy_files_df)
        error_report.update({'version': version_id})
        error_reports.append(error_report)
        bug_smells_files.to_csv(root_dir() + 'bug_smell_vectors/' + version_id + '--bug_smell_files_vectors.csv')

        smells_group_df = pandas.read_csv(root_dir() + 'pmd_results/' + smells_group_file)
        bug_smells_group_files, _ = map_by_filename(smells_group_df, buggy_files_df)
        bug_smells_group_files.to_csv(root_dir() + 'bug_smell_vectors/' + version_id + '--bug_smell_group_files_vectors.csv')

    pandas.DataFrame(error_reports).to_csv(root_dir() + 'bug_smell_vectors/0-error_reports.csv')


def map_by_filename(smells_df, buggy_files_df):
    smells_df['filename'] = smells_df['filename'].apply(lambda x: x.split('/src/main/java/')[1] if '/src/main/java/' in x else x.split('/src/test/java/')[1] if '/src/test/java/' in x else x)
    smells_df = smells_df.set_index('filename')
    smells_df['cloc_error'] = np.where(smells_df['cloc_normalization'] == smells_df['applied_cloc_normalization'], 0, 1)
    # smells_df = smells_df.drop(['cloc_normalization', 'applied_cloc_normalization'], axis=1)
    smells_df = smells_df.drop(['applied_cloc_normalization'], axis=1)

    bug_smells_comb = buggy_files_df.join(smells_df, how='left')
    bug_smells_comb['files_missing_in_src'] = np.where(bug_smells_comb['cloc_error'].isna(), 1, 0)
    bug_smells_comb['pmd_error'] = bug_smells_comb['pmd_error'].fillna(0)
    bug_smells_comb['cloc_error'] = bug_smells_comb['cloc_error'].fillna(0)
    bug_smells_comb['file_count'] = 1

    bug_smells_file_sum = bug_smells_comb.groupby(['bug_id'])[bug_smells_comb.columns.drop('bug_id')].sum().reset_index()

    error_report = {"no. bugs: ": str(len(bug_smells_file_sum)),
                    "no. bugs with files_missing_in_src: ": str(len(bug_smells_file_sum[bug_smells_file_sum['files_missing_in_src'] > 0])),
                    "no. bugs with pmd_error: ": str(len(bug_smells_file_sum[bug_smells_file_sum['pmd_error'] > 0])),
                    "no. bugs with cloc_error: ": str(len(bug_smells_file_sum[bug_smells_file_sum['cloc_error'] > 0]))}

    return bug_smells_comb, error_report


if __name__ == '__main__':
    main()
