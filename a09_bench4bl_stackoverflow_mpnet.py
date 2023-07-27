import glob

import pandas
from sentence_transformers import SentenceTransformer

from paths import root_dir


# https://huggingface.co/flax-sentence-embeddings/stackoverflow_mpnet-base


def main():
    model = SentenceTransformer('flax-sentence-embeddings/stackoverflow_mpnet-base')

    for bug_reports_csv in glob.glob(root_dir() + 'bench4bl_summary/*--bug_reports.csv'):
        version_id = bug_reports_csv.split('/')[-1].replace('--bug_reports.csv', '')
        print(version_id)
        orga, version = version_id.split('--')
        project = version.split('_')[0]
        bug_reports_df = pandas.read_csv(bug_reports_csv)
        bug_reports_df['doc'] = bug_reports_df['doc'].astype(str)
        bug_reports_df['doc_clean'] = bug_reports_df['doc_clean'].astype(str)
        bug_reports_df['duplicates'] = bug_reports_df['duplicates'].fillna('').astype(str)
        bug_reports_df['is_duplicated_by'] = bug_reports_df['is_duplicated_by'].fillna('').astype(str)

        embed(model, bug_reports_df.copy()).to_csv(root_dir() + 'stackoverflow_mpnet_embeddings/' + version_id + '--doc.csv')


def embed(model, df):
    df['text'] = df['summary'] + '\n' + df['description']
    features = model.encode(df['text'])
    df = df[['version', 'bug_id', 'is_duplicated_by', 'duplicates', 'text']]
    embdedings_df = pandas.DataFrame(features, columns=['emb_' + str(x) for x in range(0, features.shape[1])])
    return df.join(embdedings_df)


if __name__ == '__main__':
    main()
