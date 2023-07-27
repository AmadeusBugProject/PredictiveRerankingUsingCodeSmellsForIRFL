from xml.etree import ElementTree

import pandas

from paths import root_dir
from utils.bench4bl_utils import get_all_bench4bl_xml_bug_repos


def main():
    get_all_bench4bl_xml_bug_repos()
    for xml_file_name in get_all_bench4bl_xml_bug_repos():
        load_bug_reports_for_version(xml_file_name)


def load_bug_reports_for_version(xml_file):
    version = xml_file.split('/')[-5] + '--' + xml_file.split('/')[-1].rstrip('.xml')
    all_bug_reports = []
    all_bug_files = []

    et = ElementTree.parse(xml_file)
    for bug in et.getroot().findall('bug'):
        bug_id = bug.attrib['id']
        bug_information = bug.find('buginformation')
        bug_description = bug_information.find('description').text
        bug_summary = bug_information.find('summary').text

        bug_duplicates = []
        bug_is_duplicated = []
        if bug.find('links'):
            for link in bug.find('links').findall('link'):
                if link.attrib['type'] == 'Duplicate' and link.attrib['description'] == 'duplicates':
                    bug_duplicates.append(link.text)
                elif link.attrib['type'] == 'Duplicate' and link.attrib['description'] == 'is duplicated by':
                    bug_is_duplicated.append(link.text)

        all_bug_reports.append({'version': version,
                                'bug_id': bug_id,
                                'description': bug_description,
                                'summary': bug_summary,
                                'is_duplicated_by': ','.join(bug_is_duplicated),
                                'duplicates': ','.join(bug_duplicates)})

        for file_modified in bug.find('fixedFiles').findall('file'):
            all_bug_files.append({'version': version,
                                  'bug_id': bug_id,
                                  'file': file_modified.text,
                                  'mode': file_modified.attrib['type']})

    out_path = root_dir() + 'bench4bl_summary/' + version
    pandas.DataFrame(all_bug_reports).to_csv(out_path + '--bug_reports.csv')
    buggy_files_df = pandas.DataFrame(all_bug_files)
    buggy_files_df = buggy_files_df.drop_duplicates(['bug_id', 'file'], keep='first')
    buggy_files_df.to_csv(out_path + '--buggy_files.csv')


if __name__ == '__main__':
    main()
