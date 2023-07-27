import os
from subprocess import run

from paths import *
from utils.bench4bl_utils import get_all_bench4bl_versions_dirs


def main():
    bench4bl_orgas = ['Apache', 'Commomons', 'JBoss', 'Spring', 'Wildfly']
    for version in get_all_bench4bl_versions_dirs():
        print(version.split('/')[-5:-1])
        pmd_dir(version)


def pmd_dir(directory):
    ruleset = 'all_java_ruleset.xml'
    outfile = root_dir() + 'pmd_results/' + '--'.join(directory.split('/')[-5:-1]) + '.json'

    if os.path.exists(outfile):
        print('skip!')
        return

    p = run([PMD_DIR + 'bin/run.sh', 'pmd', '--no-cache', '--dir', directory, '--rulesets', ruleset, '--format', 'json', '--report-file', outfile], capture_output=True, text=True)
    with open(outfile + '.stdout', 'w') as fd:
        fd.write(p.stdout)
    with open(outfile + '.stderr', 'w') as fd:
        fd.write(p.stderr)



if __name__ == '__main__':
    main()
