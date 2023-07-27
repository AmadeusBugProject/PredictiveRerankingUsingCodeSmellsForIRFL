import glob
from pathlib import Path

import pandas

from paths import root_dir, BENCH4BL_DIR


def main():
    for orga_path in glob.glob(BENCH4BL_DIR + 'exp/*'):
        orga = orga_path.split('/')[-1]
        for project_path in glob.glob(orga_path + '/*'):
            project = project_path.split('/')[-1]
            for tool_version in glob.glob(project_path + '/*/'):
                tool_version_str = tool_version.split('/')[-2]
                tool = tool_version_str.split('_')[0]
                version = '_'.join(tool_version_str.split('_')[2:])

                out_path = root_dir() + 'bench4bl_localization_results/' + tool + '/' + orga + '/' + project + '/' + version + '/'
                Path(out_path).mkdir(parents=True, exist_ok=True)
                for txt_file in glob.glob(tool_version + 'recommended/*.txt'):
                    df = pandas.read_csv(txt_file, sep='\t', names=['rank', 'score', 'filename']).set_index('rank')
                    bug_id = txt_file.split('/')[-1].rstrip('.txt')
                    df.to_csv(out_path + bug_id + '.csv')


if __name__ == '__main__':
    main()
