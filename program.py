import sys
from pathlib import Path
import argparse
import os
import json
import random
import traceback
import re

import yaml
import pandas as pd
from tqdm import tqdm
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

from topicgpt_python.utils import APIClient, TopicTree
import topicgpt_python.assignment


def assign_topics():
    ap = argparse.ArgumentParser()
    ap.add_argument('--docs_metadata_csv', default="data_pdfs/docs_metadata.csv")
    ap.add_argument('--arxiv_topic_file', default="output/generation_arxiv_1.md")
    ap.add_argument('--arxiv_topics_yaml', default="data/misc/arxiv_topics.yaml")
    ap.add_argument('--method', choices=['vlm', 'llm'], default='vlm')
    ap.add_argument('--output_directory', '-o', required=True)
    ap.add_argument('--backend', choices=['ollama', 'openai'], default='ollama')
    ap.add_argument('--model', default='llama3.2-vision')
    ap.add_argument('--page_selection_criterion', choices=['first', 'random'], default='first')
    ap.add_argument('--num_pages', type=int, default=1,
        help="# pages to select")
    ap.add_argument('--max_page_percentage', default=80,
        help="Maximum page (amonst all pages) % to select from (this is to minimize possibility of selecting from citations pages).")
    ap.add_argument('--context', type=int, default=128_000)
    ap.add_argument('--temperature', type=float, default=0.6)
    ap.add_argument('--max_tokens', type=int, default=16_384)
    ap.add_argument('--top_p', type=float, default=0.9)
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    verbose = args.verbose

    output_dir = Path(args.output_directory)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("Opening docs metadata from: {}".format(args.docs_metadata_csv))
    docs_meta_df = pd.read_csv(args.docs_metadata_csv)
    print("# documents: {}".format(len(docs_meta_df)))

    print("Opening arxiv topics yaml: {}".format(args.arxiv_topics_yaml))
    with open(args.arxiv_topics_yaml, encoding='utf-8') as f:
        topics_conf = yaml.safe_load(f)

    topic_names = []
    for topic_d in topics_conf['topics']:
        topic_names.append(topic_d['name'])
    print("Found {} topics: {}".format(len(topic_names), topic_names))

    max_page_pct = args.max_page_percentage
    n_pages_sel = args.num_pages
    page_selection_criterion = args.page_selection_criterion

    print("Loading API client (backend: {}, model: {})".format(args.backend, args.model))
    api_client = APIClient(args.backend, args.model)
    topic_root = TopicTree().from_topic_list(args.arxiv_topic_file, from_file=True)

    method = args.method

    temperature = args.temperature
    top_p = args.top_p
    context = args.context
    max_tokens = args.max_tokens
    context_len = context - max_tokens

    if method == 'vlm':
        prompt_path = Path("prompt/assignment_img_no_examples.txt")
    elif method == 'llm':
        prompt_path = Path("prompt/assignment_no_examples.txt")
    else:
        raise ValueError("Unknown method: {}".format(method))
    print("Loading assignment prompt from: {}".format(prompt_path))
    with open(prompt_path, encoding='utf-8') as f:
        assignment_prompt = f.read()
    print("Assignment prompt:\n{}".format(assignment_prompt))

    for idoc in tqdm(range(len(docs_meta_df))):
        row = docs_meta_df.iloc[idoc]
        doc_path = Path(row['filepath'])
        print("Document [{}/{}] ({}): {}".format(idoc+1, len(docs_meta_df), doc_path, row))
        try:
            pdf_obj = pdfium.PdfDocument(str(doc_path))
            n_pages = len(pdf_obj)
            max_page = min(n_pages - 1, round((max_page_pct / 100) * n_pages - 1))
            if page_selection_criterion == 'first':
                pages = [0]
            elif page_selection_criterion == 'random':
                if max_page <= n_pages_sel - 1:
                    pages = random.choices(list(range(0, max_page+1)), k=n_pages_sel)
                else:
                    pages = random.sample(list(range(0, max_page+1)), k=n_pages_sel)
            else:
                raise ValueError("Unknown page selection criterion: {}".format(page_selection_criterion))
            print("Selected pages: {} (max page: {} ({:.02f}%) out of {} pages)".format([p+1 for p in pages], max_page+1, max_page_pct, n_pages))

            doc = None
            images = []
            for pg in pages:
                if method == 'vlm':
                    # https://github.com/pypdfium2-team/pypdfium2?tab=readme-ov-file#usage
                    bitmap = pdf_obj[pg].render(
                        scale = 1,    # 72dpi resolution
                        rotation = 0,
                        force_bitmap_format=pdfium_c.FPDFBitmap_BGRA,
                        rev_byteorder=True,
                    )
                    pg_pil = bitmap.to_pil().convert('RGB')
                    #pg_pil.save(output_dir / f"{doc_path.stem}_{pg}.jpg")
                    images.append(pg_pil)
                elif method == 'llm':
                    raise NotImplementedError("Method 'llm' not implemented")
                else:
                    raise ValueError("Unknown method: {}".format(method))
            print("Assigning topic(s)...")
            responses, prompted_docs = topicgpt_python.assignment.assignment(
                api_client=api_client,
                topics_root=topic_root,
                docs=[doc],
                assignment_prompt=assignment_prompt,
                context_len=context_len,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                verbose=verbose,
                images=[images],
                list_all_topic_candidates=True,
            )
            response = responses[0]
            if verbose:
                print(f"Response: {response}")
            # Find topic assignments from response
            matched_topics = []
            for topic_name in topic_names:
                matches = re.findall(f"\\b{topic_name}\\b", response)
                if matches:
                    matched_topics.append(topic_name)
            print("Matched topics (#={}): {}".format(len(matched_topics), matched_topics))
            breakpoint()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            traceback.print_exc()
            print("Failed to process document: {}".format(doc_path))
            print("Skipping")


def generate_arxiv_topic_file():
    ap = argparse.ArgumentParser(description="""
    Topic file needed for topic tree generation (used by TopicGPT).
    See `data/output/sample/generation_1.md` and `data/output/sample/generation_2.md` for reference.
    """)
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
                 create_docs_metadata_csv.__name__,
                 assign_topics.__name__]

    if len(sys.argv) <= 1:
        print("Missing operation (must be one of {})".format(main_funs))
        exit(1)

    fun = sys.argv[1]
    if fun not in main_funs:
        print("Invalid operation: '{}' (must be one of {})".format(fun, main_funs))
        exit(1)

    sys.argv = sys.argv[0:1] + sys.argv[2:]

    globals()[fun]()