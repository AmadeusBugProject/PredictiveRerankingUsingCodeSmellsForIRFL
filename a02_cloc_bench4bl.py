from subprocess import run

import pandas

from paths import root_dir
from utils.bench4bl_utils import get_all_bench4bl_versions_dirs


def main():
    for version in get_all_bench4bl_versions_dirs():
        print(version.split('/')[-5:-1])
        cloc_dir(version)


def cloc_dir(directory):
    outfile = root_dir() + 'cloc_results/' + '--'.join(directory.split('/')[-5:-1]) + '--java_loc_vector.csv'
    p = run(['perl', 'cloc-1.92.pl', directory, '--csv', '--by-file', '--out=' + outfile], capture_output=True, text=True)
    df = pandas.read_csv(outfile)
    df = df[~(df['language'] == 'SUM')]
    df['filename'] = df['filename'].astype(str).apply(lambda x: x.split('Bench4BL/')[1])
    df.to_csv(outfile)


if __name__ == '__main__':
    main()
