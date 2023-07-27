from pathlib import Path

import pandas

from paths import root_dir

path = root_dir() + 'p_summary_correlations/'
Path(path).mkdir(parents=True, exist_ok=True)


def main():
    smell_distance_within_file = pandas.read_csv(root_dir() + 'px_summary_dataset/smell_distances_within_bugs_averaged_by_project.csv')
    localization_performance = pandas.read_csv(root_dir() + 'p_FINAL_Bench4BL/p_summary_bootstrap/project_scores_project_wise.csv')
    classification_performance = pandas.read_csv(root_dir() + 'p_FINAL_Bench4BL/p_summary_classification_test_set/macro_average_classification_performances_per_project.csv')

    for tool in ['BLIA', 'BRTracer', 'BugLocator']:
        localization_performance[tool + '_diff'] = (localization_performance[tool + '_ext'] - localization_performance[tool]) / localization_performance[tool] * 100

    summary = pandas.merge(localization_performance, smell_distance_within_file, on='project', how='inner')
    summary = pandas.merge(summary, classification_performance, on='project', how='inner')

    target_cols = ['BLIA_diff', 'BRTracer_diff', 'BugLocator_diff']
    metrics_cols = ['jaccard_avg_op', 'hamming_avg_op', 'euclidean_avg_op', 'jaccard_avg_file', 'hamming_avg_file', 'euclidean_avg_file', 'jaccard_sum', 'hamming_sum', 'euclidean_sum', 'files_count', 'f1', 'precision', 'recall', 'rocauc',]

    correlations = []
    for metrics_col in metrics_cols:
        metrics_dict = {'metric': metrics_col}
        for target_col in target_cols:
            corr = summary[target_col].corr(summary[metrics_col], method='pearson')
            metrics_dict.update({target_col: corr})
        correlations.append(metrics_dict)
    df = pandas.DataFrame(correlations)
    df.to_csv(path + 'localization_performance_correlations.csv')
    df.to_latex(path + 'localization_performance_correlations.tex', float_format="{:0.3f}".format)


if __name__ == '__main__':
    main()
