import sys
from pathlib import Path
import argparse
import os
import json

import yaml
import pandas as pd
from tqdm import tqdm


def generate_arxiv_topic_file():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arxiv_topics_yaml', default="data/misc/arxiv_topics.yaml")
    ap.add_argument('--output', '-o', default="output/generation_arxiv_1.md")
    args = ap.parse_args()

    output_path = Path(args.output)
    if output_path.parent:
        output_path.parent.mkdir(exist_ok=True, parents=True)
    print("Output (Markdown) will be saved to: {}".format(output_path))

    with open(args.arxiv_topics_yaml, encoding='utf-8') as f:
        conf = yaml.safe_load(f)
    
    lines = []
    for topic_d in conf['topics']:
        topic_name = topic_d['name']
        topic_desc = topic_d['description']
        topic_level = topic_d.get('level', 1)
        topic_count = topic_d.get('count', 1)
        ln = f"[{topic_level}] {topic_name} (Count: {topic_count}): {topic_desc}"
        print(ln)
        lines.append(ln)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for ln in lines:
            f.write(ln + os.linesep)


def create_docs_metadata_csv():
    ap = argparse.ArgumentParser(description="""
    Kaggle arXiv dataset: https://www.kaggle.com/datasets/Cornell-University/arxiv
    Extract metadata from JSONL for sampled pdfs (in sampled_pdfs_dir), and save as csv.
    """)
    ap.add_argument('--sampled_pdfs_dir', default="data_pdfs/pdfs/")
    ap.add_argument('--kaggle_jsonl_path', default="data_pdfs/arxiv-metadata-oai-snapshot.json")
    ap.add_argument('--output_path', default="data_pdfs/docs_metadata.csv")
    args = ap.parse_args()

    print("Finding PDFs from: {}".format(args.sampled_pdfs_dir))
    sampled_pdfs = set()
    for fn in tqdm(os.listdir(args.sampled_pdfs_dir)):
        if os.path.splitext(fn)[1].lower() == ".pdf":
            sampled_pdfs.add(fn)

    print("Reading arXiv JSONL from: {}".format(args.kaggle_jsonl_path))
    with open(args.kaggle_jsonl_path, encoding='utf-8') as f:
        arxiv_meta_json_str = f.read()
    arxiv_meta_json_lines = arxiv_meta_json_str.split('\n')

    arxiv_meta_dicts = []
    print("Parsing each line into JSON")
    for line in tqdm(arxiv_meta_json_lines):
        line = line.strip()
        if line:
            arxiv_meta_dicts.append(json.loads(line))
    arxiv_meta_df = pd.DataFrame(arxiv_meta_dicts)
    arxiv_meta_df['filename'] = arxiv_meta_df['id'].str.replace('/', ' ') + ".pdf"

    print("Extracting sampled PDFs from metadata...")
    arxiv_meta_df = arxiv_meta_df[arxiv_meta_df['filename'].isin(sampled_pdfs)]
    print("Extracted {} rows".format(len(arxiv_meta_df)))

    arxiv_meta_df['filepath'] = arxiv_meta_df['filename'].map(lambda fn: os.path.join(args.sampled_pdfs_dir, fn))

    output_path = Path(args.output_path)
    print("Saving output CSV to: {}".format(output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arxiv_meta_df.to_csv(output_path)
    print("Done")


if __name__ == '__main__':
    main_funs = [generate_arxiv_topic_file.__name__,
                 create_docs_metadata_csv.__name__,]

    if len(sys.argv) <= 1:
        print("Missing operation (must be one of {})".format(main_funs))
        exit(1)

    fun = sys.argv[1]
    if fun not in main_funs:
        print("Invalid operation: '{}' (must be one of {})".format(fun, main_funs))
        exit(1)

    sys.argv = sys.argv[0:1] + sys.argv[2:]

    globals()[fun]()