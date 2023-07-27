import glob
from pathlib import Path

import pandas
from matplotlib import pyplot as plt

from paths import root_dir
from utils.stats_utils import evaluate_bootstrap, get_box

prefix_path = ''
path = root_dir() + prefix_path + 'p_summary_classification/'
Path(path).mkdir(parents=True, exist_ok=True)


def main():
    classification_stats()

def classification_stats():
    scores = pandas.DataFrame()
    for scores_path in glob.glob(root_dir() + prefix_path + 'p_model_for_smell_classification_bootstrap/**/all_groups_scores_and_model_params_for_classification_evaluation.csv'):
        scores = pandas.concat([scores, pandas.read_csv(scores_path)], ignore_index=True, )
    scores = scores.drop(columns='Unnamed: 0')

    for metric in ['f1', 'precision', 'rocauc', 'recall']:
        make_bootstrap_plot_per_smell_group(scores, metric)

    scores.groupby('metric').mean().to_csv(path + 'all_metric_means.csv')
    scores.groupby('metric').mean().T.to_latex(path + 'all_metric_means.tex', float_format="{:0.2f}".format)


def make_bootstrap_plot_per_smell_group(df, metric):
    df = df[df['metric'] == metric].copy()
    boxes = []
    bootstrap_scores = []
    groups = df.columns.drop(['metric']).to_list()
    groups.sort()
    for group in groups:
        bootstr = evaluate_bootstrap(df[group], group )
        bootstrap_scores.append(bootstr)
        boxes.append(get_box(bootstr))

    pandas.DataFrame(bootstrap_scores).to_csv(path + 'bootstrap_' + metric + '.csv')

    fig, ax1 = plt.subplots(figsize=(8, 4))
    boxplot1 = ax1.bxp(boxes, showfliers=False, widths=0.4, patch_artist=True, boxprops=dict(facecolor='lightcyan'), medianprops=dict(color="black", linewidth=1.5))
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(path + 'all_scores_' + metric + '.pdf')


if __name__ == '__main__':
    main()
