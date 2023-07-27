import glob
import json

import pandas

from paths import root_dir


def main():
    for json_filename in glob.glob(root_dir() + 'pmd_results/*.json'):
        create_smells_count_vector(json_filename)


def create_smells_count_vector(json_filename):
    smells_count_vector = []
    smells_group_count_vector = []
    pmd_errors = []

    with open(json_filename, 'r') as fd:
        print(json_filename)
        version_json = json.load(fd)
        for java_file in version_json['files']:
            smells_summary_dict = {'filename': java_file['filename'].split('Bench4BL/')[1]}
            smells_group_summary_dict = {'filename': java_file['filename'].split('Bench4BL/')[1]}
            for violation in java_file['violations']:
                increment_key(smells_summary_dict, violation['rule'])
                increment_key(smells_group_summary_dict, violation['ruleset'])

            smells_count_vector.append(smells_summary_dict)
            smells_group_count_vector.append(smells_group_summary_dict)

        for pmd_error in version_json['processingErrors']:
            errors_summary_dict = {'filename': pmd_error['filename'].split('Bench4BL/')[1], 'error': pmd_error['message'].split(':')[0]}
            pmd_errors.append(errors_summary_dict)

    outfile = json_filename.rstrip('.json')
    pandas.DataFrame(smells_count_vector).to_csv(outfile + '--smells_count_vector.csv')
    pandas.DataFrame(smells_group_count_vector).to_csv(outfile + '--smells_group_count_vector.csv')
    pandas.DataFrame(pmd_errors).to_csv(outfile + '--pmd_errors.csv')


def increment_key(dictionary, key):
    if key in dictionary.keys():
        dictionary[key] += 1
    else:
        dictionary[key] = 1


if __name__ == '__main__':
    main()
