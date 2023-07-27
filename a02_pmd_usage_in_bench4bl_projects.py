import glob

import pandas
from bs4 import UnicodeDammit

from paths import root_dir
from utils.bench4bl_utils import get_all_bench4bl_versions_dirs

path = root_dir() + 'pmd_usage_in_bench4bl_projects/'

def main():
    results = []
    for version in get_all_bench4bl_versions_dirs():
        results.append(uses_pmd(version))

    # occurrences = []
    # for version in results:
    #     occurrences.extend(version['pmd_mentions'])
    # df = pandas.DataFrame(occurrences)

    pmd_usage = []
    for version in results:
        pmd_occurrence = bool(version['pmd_mentions']) * 1
        pmd_usage.append({'orga': version['version'][0],
                          'project': version['version'][1],
                          'version': version['version'][3],
                          'pmd_occurrence': pmd_occurrence
                          })
    df = pandas.DataFrame(pmd_usage)
    df.to_csv(path + 'pmd_usage.csv')
    df = df.groupby(['orga', 'project']).agg({'pmd_occurrence': 'sum'})
    df['pmd_occurrence'] = df['pmd_occurrence'].apply(lambda x: (x > 0)*1)
    df.to_csv(path + 'pmd_usage_per_project.csv')


def uses_pmd(directory):
    results = {'version': directory.split('/')[-5:-1], 'directory': directory}
    poms = []
    poms.extend(glob.glob(directory + '*.xml'))
    poms.extend(glob.glob(directory + '*.gradle'))
    poms.extend(glob.glob(directory + '.project'))
    pmd_lines = []
    for pom in poms:
        # try:
        with open(pom, 'rb') as fd:
            dammit = UnicodeDammit(fd.read())
            bf = dammit.unicode_markup.splitlines()
            for line in bf:
                if 'pmd' in line.lower():
                    pmd_lines.append([pom, line])
        # except UnicodeDecodeError:
        #     print(pom)
    results.update({'pmd_mentions': pmd_lines})
    return results


if __name__ == '__main__':
    main()
