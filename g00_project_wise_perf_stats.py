from pathlib import Path

import pandas

from paths import root_dir
from constants import *

path = root_dir() + 'px_summary_performance/'
Path(path).mkdir(parents=True, exist_ok=True)


def main():

    multi = pandas.read_csv(root_dir() + 'p_FINAL_Bench4BL/p_summary_bootstrap/project_scores_project_wise.csv')
    hbase = pandas.read_csv(root_dir() + 'p_FINAL_HBASE/p_summary_bootstrap/project_scores_project_wise.csv')
    roo = pandas.read_csv(root_dir() + 'p_FINAL_ROO/p_summary_bootstrap/project_scores_project_wise.csv')
    camel = pandas.read_csv(root_dir() + 'p_FINAL_CAMEL/p_summary_bootstrap/project_scores_project_wise.csv')

    single_projs = pandas.concat([camel, hbase, roo])
    single_projs['pipeline'] = 'Single Project'

    comb = multi[multi['project'].isin(['ROO', 'HBASE', 'CAMEL'])].copy()
    comb['pipeline'] = 'Bench4BL'

    comb = pandas.concat([comb, single_projs])

    col_order = ['pipeline', 'project']
    tools = comb.columns.drop(['project', 'count', 'pipeline'])
    for tool_idx in list(range(0, 6, 2)):
        tool = tools[tool_idx]
        tool_ext = tool + '_ext'
        comb[tool + '_percent_change'] = (comb[tool_ext] - comb[tool])/comb[tool] * 100
        col_order += [tool, tool_ext, tool + '_percent_change']
    comb[col_order].to_latex(path + 'project_scores_comb.tex', float_format="{:0.3f}".format)


if __name__ == '__main__':
    main()
