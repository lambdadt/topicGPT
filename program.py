import sys
from pathlib import Path
import argparse
import os
import json
import random
import traceback
import re
from bdb import BdbQuit

import yaml
import pandas as pd
from tqdm import tqdm
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

from bertopic import BERTopic

from topicgpt_python.utils import APIClient, TopicTree
import topicgpt_python.assignment


def assign_topics():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--docs_metadata_csv', default="data_pdfs/docs_metadata.csv", help=".")
    ap.add_argument('--arxiv_topic_file', default="output/generation_arxiv_1.md", help=".")
    ap.add_argument('--arxiv_topics_yaml', default="data/misc/arxiv_topics.yaml", help=".")
    ap.add_argument('--method', choices=['vlm', 'llm', 'bertopic'], default='vlm', help=".")
    ap.add_argument('--output_directory', '-o', required=True, help=".")
    ap.add_argument('--backend', choices=['ollama', 'openai'], default='ollama', help=".")
    ap.add_argument('--model', default='llama3.2-vision', help=".")
    ap.add_argument('--page_selection_criterion', choices=['first', 'random'], default='first', help="If set to random, num_pages is ignored.")
    ap.add_argument('--num_pages', type=int, default=1,
        help="# pages to select")
    ap.add_argument('--max_page_percentage', default=80,
        help="Maximum page (amonst all pages) percentage to select from (this is to minimize possibility of selecting from citations pages).")
    ap.add_argument('--context', type=int, default=128_000, help=".")
    ap.add_argument('--temperature', type=float, default=0.6, help=".")
    ap.add_argument('--max_tokens', type=int, default=16_384, help=".")
    ap.add_argument('--top_p', type=float, default=0.9, help=".")
    ap.add_argument('--shuffle_topics', action='store_true',
    help="Shuffle topic order (by shuffling the lines of arxiv_topic_file) for every prompt.")
    ap.add_argument('--continue_from_index', type=int, default=1,
    help="Continue processing from this index (1-based). Based on docs_metadata_csv. Also see log to determine where the process was dropped off.")
    ap.add_argument('--verbose', action='store_true', help=".")
    args = ap.parse_args()

    verbose = args.verbose

    output_dir = Path(args.output_directory)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("Output directory: {}".format(output_dir))

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

    method = args.method

    temperature = args.temperature
    top_p = args.top_p
    context = args.context
    max_tokens = args.max_tokens
    context_len = context - max_tokens

    shuffle_topics = args.shuffle_topics

    individual_results_save_dir = output_dir / "result_jsons"
    individual_results_save_dir.mkdir(exist_ok=True, parents=True)

    print(f"Method: {method}; temperature: {temperature}; top_p: {top_p}; "
          f"context: {context}; max_tokens: {max_tokens}; context_len: {context_len}; shuffle_topics: {shuffle_topics}")

    prompt_path = None
    assignment_prompt = None
    if method == 'vlm':
        prompt_path = Path(Path(__file__).parent, "prompt/assignment_img_no_examples.txt")
    elif method == 'llm':
        prompt_path = Path(Path(__file__).parent, "prompt/assignment_no_examples.txt")
    elif method == 'bertopic':
        pass
    else:
        raise ValueError("Unknown method: {}".format(method))
    
    if prompt_path is not None:
        print("Loading assignment prompt from: {}".format(prompt_path))
        with open(prompt_path, encoding='utf-8') as f:
            assignment_prompt = f.read()
        print("Assignment prompt:\n{}".format(assignment_prompt))

    continue_from_idx = max(0, args.continue_from_index - 1)
    print("Continuing from index: {} (/{})".format(continue_from_idx + 1, len(docs_meta_df)))

    all_results = []
    err_results = []
    for idoc in tqdm(range(len(docs_meta_df))):
        if idoc < continue_from_idx:
            continue
        row = docs_meta_df.iloc[idoc]
        doc_path = Path(row['filepath'])
        print("\n" + (75 * "="))
        print("Document [{}/{}] ({}): Title: {}; Topics: {} (#errs={})".format(idoc+1, len(docs_meta_df), doc_path,row['title'], row['topics'], len(err_results)))
        topic_root = TopicTree().from_topic_list(args.arxiv_topic_file, from_file=True, shuffle_lines=shuffle_topics)

        res_save_path = individual_results_save_dir / (Path(row['filepath']).stem + ".json")
        try:
            pdf_obj = pdfium.PdfDocument(str(doc_path))
            n_pages = len(pdf_obj)
            max_page = min(n_pages - 1, round((max_page_pct / 100) * n_pages - 1))
            if page_selection_criterion == 'first':
                pages = [0]
            elif page_selection_criterion == 'random':
                if n_pages_sel <= max_page + 1:
                    pages = random.sample(list(range(0, max_page+1)), k=n_pages_sel)
                else:
                    pages = random.choices(list(range(0, max_page+1)), k=n_pages_sel)
            else:
                raise ValueError("Unknown page selection criterion: {}".format(page_selection_criterion))
            print("Selected page(s): {} (max page: {} ({:.02f}%) out of {} pages)".format([p+1 for p in pages], max_page+1, max_page_pct, n_pages))

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
                elif method == 'llm' or method == 'bertopic':
                    textpage = pdf_obj[pg].get_textpage()
                    text_all = textpage.get_text_bounded()
                    doc = text_all if not doc else (doc + "\n\n" + text_all)
                else:
                    raise ValueError("Unknown method: {}".format(method))
            if method == 'vlm' or method == 'llm':
                print("Assigning topic(s)...")
                responses, prompted_docs, cur_prompts = topicgpt_python.assignment.assignment(
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
                    return_prompts=True,
                )
                response = responses[0]
                cur_prompt = cur_prompts[0]
                if verbose:
                    print(f"Response: {response}")
                # Find topic assignments from response
                matched_topics = []
                for topic_name in topic_names:
                    matches = re.findall(f"\\b{topic_name}\\b", response)
                    if matches:
                        matched_topics.append(topic_name)
                print("Matched topics (#={}): {}".format(len(matched_topics), matched_topics))
            elif method == 'bertopic':
                cur_prompt = None
                response = None
                matched_topics = None
            else:
                raise AssertionError()
            res = {
                **row.to_dict(),
                'prompt': cur_prompt,
                'response': response,
                'matched_topics': matched_topics,
                'document_text': doc,
                'pages': pages,
            }
            if method == 'vlm' or method == 'llm':
                with open(res_save_path, 'w', encoding='utf-8') as f:
                    json.dump(res, f, ensure_ascii=False, indent=2)
                print("Saved results to: {}".format(res_save_path))
            all_results.append(res)
        except (KeyboardInterrupt, BdbQuit):
            exit(1)
        except Exception as e:
            err_d = {
                **row.to_dict(),
                'error': str(e),
            }
            with open(res_save_path, 'w', encoding='utf-8') as f:
                json.dump(err_d, f, ensure_ascii=False, indent=2)
            err_results.append(err_d)
            traceback.print_exc()
            print("Failed to process document: {}".format(doc_path))
            print("Skipping")
    
    if method == 'bertopic':
        print("Assigning topics using BERTopic...")
        doclist = []
        for res in all_results:
            doclist.append(res['document_text'])
        bertopic_model = BERTopic()
        bertopic_topics, bertopic_probs = bertopic_model.fit_transform(doclist)
        for res, topic in zip(all_results, bertopic_topics):
            res['matched_topics'] = [topic]

    all_results_save_path = output_dir / "all_results.json"
    err_results_save_path = output_dir / "failed_results.json"
    print("Saving results (#={}) to: {}".format(len(all_results), all_results_save_path))
    with open(all_results_save_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print("Saving err results (#={}) to: {}".format(len(err_results), err_results_save_path))
    with open(err_results_save_path, 'w', encoding='utf-8') as f:
        json.dump(err_results, f, ensure_ascii=False, indent=2)
    print("Done")


def generate_arxiv_topic_file():
    ap = argparse.ArgumentParser(description="""
    Topic file needed for topic tree generation (used by TopicGPT).
    See `data/output/sample/generation_1.md` and `data/output/sample/generation_2.md` for reference.
    """, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument('--arxiv_topics_yaml', default="data/misc/arxiv_topics.yaml", help=".")
    ap.add_argument('--output', '-o', default="output/generation_arxiv_1.md", help=".")
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
    """, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--sampled_pdfs_dir', default="data_pdfs/pdfs/", help="Where the sampled arXiv PDFs are stored. See <https://github.com/lambdadt/DocImageAnalysis>.")
    ap.add_argument('--kaggle_jsonl_path', default="data_pdfs/arxiv-metadata-oai-snapshot.json", help="From Kaggle arXiv dataset: https://www.kaggle.com/datasets/Cornell-University/arxiv")
    ap.add_argument('--arxiv_topics_yaml', default="data/misc/arxiv_topics.yaml", help=".")
    ap.add_argument('--output_path', default="data_pdfs/docs_metadata.csv", help=".")
    ap.add_argument('--sample', type=int, default=-1, help="If set to value greater than 0, random documents will be sampled for that amount instead of using the entire collection.")
    ap.add_argument('--balanced_sampling', default='none', choices=['none', 'category', 'topic'],
        help="Will sample a balanced number for each category (or topic; category is more fine-grained). Output may not be equal to --sample depending on amount of overlap present in topic distribution.")
    ap.add_argument('--strict_topic_extraction', action='store_true')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    print("Reading arXiv topic YAML from: {}".format(args.arxiv_topics_yaml))
    with open(args.arxiv_topics_yaml, encoding='utf-8') as f:
        conf = yaml.safe_load(f)
    prefix_to_topic = {}
    for topic_d in conf['topics']:
        for prefix in topic_d['prefixes']:
            prefix_to_topic[prefix] = topic_d['name']

    print("Finding PDFs from: {}".format(args.sampled_pdfs_dir))
    sampled_pdfs = set()
    for fn in tqdm(os.listdir(args.sampled_pdfs_dir)):
        if os.path.splitext(fn)[1].lower() == ".pdf":
            sampled_pdfs.add(fn)
    
    if args.sample > 0 and args.balanced_sampling == 'none':
        print("Randomly sampling {} documents out of {}.".format(args.sample, len(sampled_pdfs)))
        sampled_pdfs = random.sample(sorted(sampled_pdfs), k=args.sample)
        sampled_pdfs = set(sampled_pdfs)

    print("Reading arXiv JSONL from: {}".format(args.kaggle_jsonl_path))
    with open(args.kaggle_jsonl_path, encoding='utf-8') as f:
        arxiv_meta_json_str = f.read()
    arxiv_meta_json_lines = arxiv_meta_json_str.split('\n')

    arxiv_meta_dicts = []
    print("Parsing each line as JSON")
    for iline, line in enumerate(tqdm(arxiv_meta_json_lines)):
        line = line.strip()
        if line:
            d = json.loads(line)
            d['topics'] = []
            categories = d['categories'].split(' ')
            if args.strict_topic_extraction:
                categories = [c for c in categories if c not in ['adap-org', 'chao-dyn', 'patt-sol']] # There are more that are not listed in https://arxiv.org/category_taxonomy
            for cate in categories:
                for prefix, topic in prefix_to_topic.items():
                    if cate.startswith((prefix + ".") if '.' in cate else prefix):
                        d['topics'].append(topic)
                        break
            if args.strict_topic_extraction and len(categories) != len(d['topics']):
                print(f"{categories=} {d['topics']=}")
                breakpoint()
            if args.verbose:
                print(f"[{iline+1}/{len(arxiv_meta_json_lines)}] Topics: {d['topics']}; Categories: {d['categories']}")
            d['topics'] = json.dumps(d['topics'])
            arxiv_meta_dicts.append(d)
    arxiv_meta_df = pd.DataFrame(arxiv_meta_dicts)
    arxiv_meta_df['filename'] = arxiv_meta_df['id'].str.replace('/', ' ') + ".pdf"

    print("Extracting sampled PDFs from metadata...")
    arxiv_meta_df = arxiv_meta_df[arxiv_meta_df['filename'].isin(sampled_pdfs)]
    print("Extracted {} rows".format(len(arxiv_meta_df)))

    arxiv_meta_df['filepath'] = arxiv_meta_df['filename'].map(lambda fn: os.path.join(args.sampled_pdfs_dir, fn))

    # Balanced sampling based on categories
    if args.sample > 0 and args.balanced_sampling != 'none':
        print("Sampling based on categories; target: {}".format(args.sample))
        categories = set()
        for d in arxiv_meta_dicts:
            if args.balanced_sampling == 'category':
                categories.update(d['categories'].split(' '))
            elif args.balanced_sampling == 'topic':
                categories.update(json.loads(d['topics']))
            else:
                raise ValueError()
        print("Categories (#={}): {}".format(len(categories), categories))
        n_sample_per_cate = round(1.5 * (args.sample / len(categories)))
        arxiv_meta_df_sampled_list = []
        for cate in categories:
            mask_col = 'categories' if args.balanced_sampling == 'category' else 'topics'
            arxiv_meta_df_cate = arxiv_meta_df[arxiv_meta_df[mask_col].str.contains(cate)]
            sampled_indices = random.sample(list(range(len(arxiv_meta_df_cate))), k=min(len(arxiv_meta_df_cate), n_sample_per_cate))
            arxiv_meta_df_sampled_list.append(arxiv_meta_df_cate.iloc[sampled_indices])
        arxiv_meta_df = pd.concat(arxiv_meta_df_sampled_list, axis=0)
        arxiv_meta_df = arxiv_meta_df.drop_duplicates(subset=['filepath'])
        print("# entries after balanced sampling: {}".format(len(arxiv_meta_df)))

    output_path = Path(args.output_path)
    print("Saving output CSV to: {}".format(output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arxiv_meta_df.to_csv(output_path, index=False)
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