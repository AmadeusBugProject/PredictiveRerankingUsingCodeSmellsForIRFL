# Predictive Reranking using Code Smells for Information Retrieval Fault Localization
This repository constitutes the source code and results for the paper "Predictive Reranking using Code Smells for Information Retrieval Fault Localization", by Thomas Hirsch and Birgit Hofer, 2023.

## Preliminaries
This repository contains the Python code of our experiments and all results used in our paper.
The underlying dataset and intermediate files are available on [Zenodo](https://zenodo.org/record/8186775), or can be created from scratch by importing Bench4BL (see details below).

### Python environment
  - python=3.8
  - pandas
  - numpy
  - joblib
  - scikit-learn==1.0.2
  - keras
  - tensorflow
  - nltk
  - sentence-transformers
  - matplotlib
  - seaborn

Conda files are located in the root directory of the repository, [conda_from_history.yml](conda_from_history.yml).

### Bench4BL
The [Bench4BL](https://github.com/exatoa/Bench4BL) dataset was used in our evaluation.
All data necessary for our machine learning and localization experiments is available on [Zenodo](https://zenodo.org/record/8186775).

However, if the data is to be re-imported and recalculated directly from Bench4BL:
Bench4BL has to be downloaded and paths to the benchmark root set accordingly in [paths.py](paths.py).
BugLocator, BRTracer, and BLIA have to be run on the Bench4BL dataset using the scripting provided by the benchmark.
PMD has to be installed in version 6.45.0 and path to PMD set accordingly in [paths.py](paths.py).


# Structure of this repository
## General utility functions

- [constants.py](constants.py) Contains parameters for the NN model, and other parameters.
- [paths.py](paths.py) Contains paths to external datasources and tools, e.g. Bench4BL and PMD.

- [utils/bench4bl_utils.py](utils/bench4bl_utils.py) Helper functions for resolving paths to Bench4BL benchmark.
- [utils/dataset_utils.py](utils/dataset_utils.py) Helper functions for loading datasets and performing dataset splits.                                      
- [utils/Logger.py](utils/Logger.py) Logging.
- [utils/nn_classifier.py](utils/nn_classifier.py) NN classifier model.
- [utils/scoring_utils.py](utils/scoring_utils.py) Metrics.
- [utils/stats_utils.py](utils/stats_utils.py) Wrapper methods for statistical tests.


## Experiment
### Dataset setup and preparation
The following scripts are responsible to create, import, and set up data that is used in our experiments.
The produced data is already part of this repository, the execution of these scripts is therefore only necessary when data is to be re-imported from the Bench4BL repository.
- [a00_pmd_bench4bl.py](a00_pmd_bench4bl.py) Runs PMD on all projects and versions contained in the Bench4BL dataset. The utilized ruleset is defined in [all_java_ruleset.xml](all_java_ruleset.xml). Results are stored in [pmd_results](pmd_results).
- [a01_vectorize_pmd.py](a01_vectorize_pmd.py) Creates csv vectors from PMD outputs. 
- [a02_cloc_bench4bl.py](a02_cloc_bench4bl.py) Runs cloc on all projects and versions contained in the Bench4BL dataset. Only Java files are considered. Results are stored in [cloc_results](cloc_results).
- [a02_pmd_usage_in_bench4bl_projects.py](a02_pmd_usage_in_bench4bl_projects.py) Searches for occurrence of PMD in the build files of all projects and versions contained in the Bench4BL dataset. Results are stored in [pmd_usage_in_bench4bl_projects](pmd_usage_in_bench4bl_projects).
- [a03_import_bugs_from_bench4bl.py](a03_import_bugs_from_bench4bl.py) Imports textual bug reports and corresponding fixed files ground truth from Bench4BL. Results are stored in [bench4bl_summary](bench4bl_summary).
- [a04_normalize_smells_by_loc.py](a04_normalize_smells_by_loc.py) Normalizes the smell vectors for each file with its LOC count. Results are stored in [pmd_results](pmd_results).
- [a05_bench4bl_file_features.py](a05_bench4bl_file_features.py) Creates feature vectors for each bug report from PMD smells. Results are stored in [bug_smell_vectors](bug_smell_vectors). 
- [a05_bench4bl_ranking_results.py](a05_bench4bl_ranking_results.py) Imports the results of BugLocator, BRTracer, and BLIA from the Bench4BL benchmark. Results are stored in [bench4bl_localization_results](bench4bl_localization_results).
- [a09_bench4bl_stackoverflow_mpnet.py](a09_bench4bl_stackoverflow_mpnet.py) Creates document embeddings for all textual bug reports using the [stackoverflow_mpnet-base](https://huggingface.co/flax-sentence-embeddings/stackoverflow_mpnet-base) model. Results are stored in [stackoverflow_mpnet_embeddings](stackoverflow_mpnet_embeddings).

### Preliminary experiments and dataset splitting
The following scripts perform data set splitting, and the preliminary experiments used for feature selection as discussed in Section VI of the paper.
- [b00_analyze_most_promising_smells.py](b00_analyze_most_promising_smells.py) Assumes a perfect smell oracle (by using the known smells of the ground truth files) and applies it to rerank the IRFL tools outputs on the Classification Training Set (the older half of versions in the dataset). Then evaluates the achievable localization performance increase for each smell group. Results are stored in [h_analyze_most_promising_smells](h_analyze_most_promising_smells).
- [c00_make_dataset_splits_bootstrap.py](c00_make_dataset_splits_bootstrap.py) Performs dataset splitting. Splits are performed on a temporal ordering of versions of each contained software project. Data is greedily split into 50/25/25, resulting in a Classification training set (used for NN model training), a ranking training set (used to estimate weights for linear combination of smell distance and IRFL suspicousness scores), and a test set (used for evaluating the localization performance achievable by our pipeline). Bootstrapping is applied by resampling fractions of 0.8, 20 times, resulting in 20 sets of the three splits to be used in the following eperiments. Datasets are stored in [p_FINAL_Bench4BL/p_model_for_smell_classification_bootstrap](p_FINAL_Bench4BL/p_model_for_smell_classification_bootstrap) for the full Bench4BL dataset, for the single project experiments please refer to p_FINAL_CAMEL, p_FINAL_HBASE, and p_FINAL_ROO accordingly.
- [c01_full_dataset_stats.py](c01_full_dataset_stats.py) Calculates various statistics on the created datasets. Results are stored in [px_summary_dataset](px_summary_dataset).
- [d00_model_for_smell_classification_performance_all_groups_bootstrap.py](d00_model_for_smell_classification_performance_all_groups_bootstrap.py) Trains a NN model on the Classification data set and evaluates its classification performance on the Ranking training set. This is performed for all 20 bootstrap iterations, the resulting data is stored in [p_FINAL_Bench4BL/p_model_for_smell_classification_bootstrap](p_FINAL_Bench4BL/p_model_for_smell_classification_bootstrap)  for the full Bench4BL dataset, for the single project experiments please refer to p_FINAL_CAMEL, p_FINAL_HBASE, and p_FINAL_ROO accordingly.
- [d01_classification_performance_evaluation_for_feature_selection_preliminariy_all_smell_groups.py](d01_classification_performance_evaluation_for_feature_selection_preliminariy_all_smell_groups.py) Creates summary and bootstrap statistics from the previous step. Results are stored in [p_FINAL_Bench4BL/p_summary_classification](p_FINAL_Bench4BL/p_summary_classification) for the full Bench4BL dataset.

### Localization experiments
The following scripts perform our actual localization experiments. These scripts are applied to bootstrapped dataset splits. Results are stored in [p_FINAL_Bench4BL](p_FINAL_Bench4BL) for the full Bench4BL dataset, for the single project experiments please refer to p_FINAL_CAMEL, p_FINAL_HBASE, and p_FINAL_ROO accordingly. For a detailed experiment setup we refer to our paper.
- [e01_model_for_localization_bootstrap.py](e01_model_for_localization_bootstrap.py) Trains NN models for smell classification and performs predictions on the corresponding test sets. Results are stored in [p_FINAL_Bench4BL/p_model_for_smell_classification_bootstrap](p_FINAL_Bench4BL/p_model_for_smell_classification_bootstrap).
- [e02_ranking_training_bootstrap.py](e02_ranking_training_bootstrap.py) Performs reranking of the Ranking training set based on predicted smells by the model created in the previous step. Score combination is performed by linear combination of IRFL tools suspicousness scores and smell distances calculated based on our predictions. Results are stored in [p_FINAL_Bench4BL/p_ranking_training_bootstrap](p_FINAL_Bench4BL/p_ranking_training_bootstrap).
- [e03_get_best_weigths_per_project_bootstrap.py](e03_get_best_weigths_per_project_bootstrap.py) Evaluates the outputs of the previous steps in order to pick the best weights for each project and IRFL tool. Results are stored in [p_FINAL_Bench4BL/p_ranking_training_bootstrap](p_FINAL_Bench4BL/p_ranking_training_bootstrap).
- [e04_ranking_test_project_wise_bootstrap.py](e04_ranking_test_project_wise_bootstrap.py) Uses predictions of the final NN model and the weights obtained from the previous step to perform rerankings on the Test set. Results are stored in [p_FINAL_Bench4BL/p_ranking_test_proejct_wise_bootstrap](p_FINAL_Bench4BL/p_ranking_test_proejct_wise_bootstrap).

### Result collection and evaluation
The following scripts calculate final scores and statistics from the 20 bootstrap iterations of the previous block of scripts.
- [f00_bootstrap_summary_compare_map_and_ttest.py](f00_bootstrap_summary_compare_map_and_ttest.py) Calculates localization performance using the MAP metric and performs statistical tests. Results are stored in [p_FINAL_Bench4BL/p_summary_bootstrap](p_FINAL_Bench4BL/p_summary_bootstrap).
- [f01_boostrap_summary_compare_classification_performance.py](f01_boostrap_summary_compare_classification_performance.py) Calculates classifier performance of our final model. Results are stored in [p_FINAL_Bench4BL/p_summary_bootstrap](p_FINAL_Bench4BL/p_summary_bootstrap).
- [f02_bootstrap_summary_model_classification_performance_eval_test_set_for_all_projects.py](f02_bootstrap_summary_model_classification_performance_eval_test_set_for_all_projects.py) Calculates classifier performance and project wise classifier performance of our final model. Results are stored in [p_FINAL_Bench4BL/p_summary_classification_test_set](p_FINAL_Bench4BL/p_summary_classification_test_set).

### Further analysis
The following scripts collect statistics and results to create latex tables and additional analysis used in our paper.
- [g00_project_wise_perf_stats.py](g00_project_wise_perf_stats.py) Creates overview latex table comparing the Bench4BL and single project trained pipelines. Results are stored in [px_summary_performance](px_summary_performance).
- [g01_project_multiple_file_smell_distances.py](g01_project_multiple_file_smell_distances.py) Analyses smell distances within each bug's ground truth files. Results are stored in [px_summary_dataset](px_summary_dataset).
- [g02_performance_correlation_analysis.py](g02_performance_correlation_analysis.py) Performs correlation analysis of our pipeline's MAP localization performance, classification performance, and smell distance measures from previous step. Results are stored in [p_FINAL_Bench4BL/p_summary_correlations](p_FINAL_Bench4BL/p_summary_correlations) for the full Bench4BL dataset.

## Results
- [pmd_catalogue/all_smells.json](pmd_catalogue/all_smells.json) lists all PMD smells and associated groups that occur in the dataset.
- [px_summary_dataset](px_summary_dataset) contains statistics and information about the utilized dataset.

Results for preliminary experiments for feature selection:
- [h_analyze_most_promising_smells](h_analyze_most_promising_smells) contains the results of our preliminary experiment into each smell group's information content towards localization.
- [p_FINAL_Bench4BL/p_summary_classification](p_FINAL_Bench4BL/p_summary_classification) contains the results of our preliminary experiments into the classifiability of smell groups from textual bug reports.

Results of our localization eperiments:
- [p_FINAL_Bench4BL/p_summary_bootstrap](p_FINAL_Bench4BL/p_summary_bootstrap) contains the results of our localization experiments, project wise MAP performance summary can be found in [p_FINAL_Bench4BL/p_summary_bootstrap/project_scores_tool_wise_others.tex](p_FINAL_Bench4BL/p_summary_bootstrap/project_scores_tool_wise_others.tex).
- [p_FINAL_Bench4BL/p_summary_classification_test_set](p_FINAL_Bench4BL/p_summary_classification_test_set) contains the results of our classification performance analysis of the predictions used in localization. A project wise classification performance summary can be found in [p_FINAL_Bench4BL/p_summary_classification_test_set/macro_average_classification_performances_per_projectother.tex](p_FINAL_Bench4BL/p_summary_classification_test_set/macro_average_classification_performances_per_projectother.tex)

## Licence
All code and results are licensed under [AGPL v3](https://www.gnu.org/licenses/agpl-3.0.html.en), according to LICENSE file.
Other licences may apply for some tools and datasets contained in this repo: [cloc-1.92.pl](https://github.com/AlDanial/cloc) under [GPL v2](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html), and data originating from [Bench4BL](https://github.com/exatoa/Bench4BL) under [CCA 4.0](https://creativecommons.org/licenses/by/4.0/).
