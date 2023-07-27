import glob

from paths import BENCH4BL_DIR


def get_all_bench4bl_versions_dirs():
    versions_dirs = []
    bench4bl_orgas = ['Apache', 'Commons', 'JBoss', 'Spring', 'Wildfly']
    for orga in bench4bl_orgas:
        for project in glob.glob(BENCH4BL_DIR + 'data/' + orga + '/*/'):
            for version in glob.glob(project + 'sources/*/'):
                versions_dirs.append(version)
    return versions_dirs


def get_all_bench4bl_xml_bug_repos():
    xml_bug_repos = []
    bench4bl_orgas = ['Apache', 'Commons', 'JBoss', 'Spring', 'Wildfly']
    for orga in bench4bl_orgas:
        for project in glob.glob(BENCH4BL_DIR + 'data/' + orga + '/*/'):
            for xml_bug_repo in glob.glob(project + 'bugrepo/repository/*.xml'):
                print(xml_bug_repo)
                xml_bug_repos.append(xml_bug_repo)
    return xml_bug_repos


def rename_java_file(jf):
    jf = '/' + jf
    if '/java/' in jf:
        return jf.split('/java/')[1].replace('/', '.')
    if '/groovy/' in jf:
        return jf.split('/groovy/')[1].replace('/', '.')
    if '/org/' in jf:
        return 'org.' + jf.split('/org/')[1].replace('/', '.')
    if '/com/' in jf:
        return 'com.' + jf.split('/com/')[1].replace('/', '.')
    # print(jf)
    return jf