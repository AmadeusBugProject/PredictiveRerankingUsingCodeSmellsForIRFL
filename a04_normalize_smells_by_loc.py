import glob

import pandas

from paths import root_dir


def main():
    for smells_count_file in (glob.glob(root_dir() + 'pmd_results/*--smells_count_vector.csv') + glob.glob(root_dir() + 'pmd_results/*--smells_group_count_vector.csv')):
        full_version_name = smells_count_file.replace('--smells_count_vector.csv', '').replace('--smells_group_count_vector.csv', '').split('/')[-1]
        print(full_version_name)

        pmd_error_file = smells_count_file.replace('--smells_count_vector.csv', '').replace('--smells_group_count_vector.csv', '') + '--pmd_errors.csv'
        pmd_error_df = pandas.read_csv(pmd_error_file)
        pmd_error_df = pmd_error_df.drop('Unnamed: 0', axis=1)
        if len(pmd_error_df):
            pmd_error_df['pmd_error'] = 1
            pmd_error_df = pmd_error_df[['filename', 'pmd_error']]
        else:
            pmd_error_df = pandas.DataFrame(columns=['filename', 'pmd_error'])
        pmd_error_df = pmd_error_df.set_index('filename')

        cloc_result_file = root_dir() + 'cloc_results/' + full_version_name + '--java_loc_vector.csv'
        cloc_df = pandas.read_csv(cloc_result_file)
        cloc_df = cloc_df.drop('Unnamed: 0', axis=1)
        cloc_df = cloc_df[cloc_df['language'] == 'Java']

        # normalization factor is total lenght except empty lines (could be modified to code only)
        cloc_df['cloc_normalization'] = cloc_df['code'] + cloc_df['comment']
        cloc_mean = cloc_df['cloc_normalization'].mean()
        cloc_df = cloc_df[['filename', 'cloc_normalization']]
        cloc_df = cloc_df.set_index('filename')

        pmd_df = pandas.read_csv(smells_count_file)
        pmd_df = pmd_df.drop('Unnamed: 0', axis=1)
        pmd_df = pmd_df.fillna(0)
        pmd_df = pmd_df.set_index('filename')

        pmd_df = pmd_df.join(cloc_df, how='outer')
        pmd_df = pmd_df.join(pmd_error_df, how='outer')

        # this really should not occur, but better safe then sorry (since this takes a long time to run):
        pmd_df['applied_cloc_normalization'] = pmd_df['cloc_normalization'].fillna(cloc_mean)
        pmd_df['applied_cloc_normalization'] = pmd_df['applied_cloc_normalization'].replace({0: 1})

        for col in pmd_df.columns.drop(['pmd_error', 'cloc_normalization', 'applied_cloc_normalization']):
            pmd_df[col] = pmd_df[col]/pmd_df['applied_cloc_normalization']

        pmd_df.to_csv(smells_count_file.rstrip('.csv') + '--cloc_normalized.csv')


if __name__ == '__main__':
    main()
