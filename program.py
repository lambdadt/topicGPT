import sys
from pathlib import Path
import argparse
import os

import yaml


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


if __name__ == '__main__':
    main_funs = [generate_arxiv_topic_file.__name__]

    if len(sys.argv) <= 1:
        print("Missing operation (must be one of {})".format(main_funs))
        exit(1)

    fun = sys.argv[1]
    if fun not in main_funs:
        print("Invalid operation: '{}' (must be one of {})".format(fun, main_funs))
        exit(1)

    sys.argv = sys.argv[0:1] + sys.argv[2:]

    globals()[fun]()