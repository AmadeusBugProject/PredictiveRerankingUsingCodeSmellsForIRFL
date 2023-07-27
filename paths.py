import pathlib


def root_dir():
    return str(pathlib.Path(__file__).parent) + '/'

BENCH4BL_DIR = root_dir() + '../informationRetrieval/Bench4BL/'
PMD_DIR = root_dir() + '../pmd-bin-6.45.0/'
