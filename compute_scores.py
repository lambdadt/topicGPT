import os
import sys
from pathlib import Path
import json
import re
from collections import Counter
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from cdlib import evaluation, NodeClustering

print("Importing: from topicgpt_python.utils import calculate_metrics")
from topicgpt_python.utils import calculate_metrics


ap = argparse.ArgumentParser()
ap.add_argument('--output_dir', '-o', default="output/results/")
args = ap.parse_args()

output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)
print(f"Output dir: {output_dir}")


# Ported from analyse_outputs.ipynb

docs_meta_csv_path = Path("data_pdfs/docs_metadata.csv")
docs_meta_df = pd.read_csv(docs_meta_csv_path)
print(f"Docs meta dataframe:\n{docs_meta_df}")

with open("data/misc/arxiv_topics.yaml") as f:
    topics_conf = yaml.safe_load(f)
topic_names = []
for d in topics_conf['topics']:
    topic_names.append(d['name'])
print(f"{topic_names=}")

assig_base_dir = Path("output/topic_assignments/20250417T203941/")
assig_outputs_all = {
    'bertopic_first_page': {
        'path': assig_base_dir / "bertopic_first_page/all_results.json"
    },
    'bertopic_random_page': {
        'path': assig_base_dir / "bertopic_random_page/all_results.json"
    },
    'bertopic_2_random_pages': {
        'path': assig_base_dir / "bertopic_2_random_pages/all_results.json"
    },
    'vlm_first_page': {
        'path': assig_base_dir / "vlm_first_page/all_results.json",
    },
    'vlm_random_page': {
        'path': assig_base_dir / "vlm_random_page/all_results.json",
    },
    'vlm_2_random_pages': {
        'path': assig_base_dir / "vlm_2_random_pages/all_results.json",
    },
    
    'llm_first_page': {
        'path': assig_base_dir / "llm_first_page/all_results.json",
    },
    'llm_random_page': {
        'path': assig_base_dir / "llm_random_page/all_results.json",
    },
    'llm_2_random_pages': {
        'path': assig_base_dir / "llm_2_random_pages/all_results.json",
    },
}
for k, assig_d in assig_outputs_all.items():
    assert os.path.isfile(assig_d['path']), "Invalid path: {}".format(assig_d['path'])


def identify_matched_topics_from_response(response: str, topic_names: list):
    keywords = ["Final Answer", "Final answer", "final answer", "Conclusion", "In conclusion"]
    for kw in keywords:
        if kw in response:
            response = response[response.index(kw)+len(kw):]
            break
    output_pattern = r"\[(?:\d+)\] ([^:]+): (?:.+)"
    output_topics = re.findall(output_pattern, response)
    output_topics = [t for t in output_topics if t in topic_names]
    if len(output_topics) == 0:
        found_topics = []
        for topic_name in topic_names:
            if topic_name in response:
                found_topics.append((response.index(topic_name), topic_name))
        for _, topic_name in sorted(found_topics):
            output_topics.append(topic_name)
    return output_topics


results_allmodels = []

pick1topic_sorted_alphabetically = True
for imodel, (model_name, assig_output_d) in enumerate(assig_outputs_all.items()):
    print("Model {}/{}: {}".format(imodel+1, len(assig_outputs_all), model_name))
    with open(assig_output_d['path'], encoding='utf-8') as f:
        assig_datas = json.load(f)
    curmodel_assig_datas_fordf = []
    true_topics_unique = set()
    pred_topics_unique = set()
    gt_pred_topic_names_match = "bertopic" not in model_name.lower()
    for idoc, assig_d in enumerate(tqdm(assig_datas)):
        true_topics = json.loads(assig_d['topics'])
        llm_response = assig_d.get('response', None)
        if llm_response is not None:
            pred_topics = identify_matched_topics_from_response(assig_d['response'], topic_names=topic_names)
        else:
            pred_topics = assig_d['matched_topics']
        if pick1topic_sorted_alphabetically:
            true_topics = sorted(set(true_topics))
            pred_topics = sorted(set(pred_topics))
        curmodel_assig_datas_fordf.append({
            'gt': true_topics[0],
            'pred': pred_topics[0] if len(pred_topics) > 0 else "XXXX",
            'gt_all': true_topics,
            'pred_all': pred_topics,
        })
        true_topics_unique.update(true_topics)
        pred_topics_unique.update(pred_topics)

    both_included_topics_unique = true_topics_unique.intersection(pred_topics_unique)
    for i, d in enumerate(curmodel_assig_datas_fordf):
        true_topics_ix = [t for t in d['gt_all'] if t in both_included_topics_unique]
        pred_topics_ix = [t for t in d['pred_all'] if t in both_included_topics_unique]
        d['gt_ix'] = true_topics_ix or ['DUMMY']
        if i == 0:
            d['gt_ix'].append('DUMMY') # Need at least 1 DUMMY in gt_ix
        d['pred_ix'] = pred_topics_ix or ['DUMMY']

    cur_model_assigdf = pd.DataFrame(curmodel_assig_datas_fordf)
    harmonic_purity, ari, mis = calculate_metrics('gt', 'pred', cur_model_assigdf)

    results_d = {
        'name': model_name,
        'harmonic_purity': harmonic_purity,
        'ari': ari,
        'mis': mis,
    }

    if gt_pred_topic_names_match:
        print("Creating node clusterings")
        gt_nc = NodeClustering(communities=cur_model_assigdf['gt_all'].tolist(), graph=None, method_name="ground_truth")
        pred_nc = NodeClustering(communities=cur_model_assigdf['pred_all'].tolist(), graph=None, method_name="prediction")
        gt_ix_nc = NodeClustering(communities=cur_model_assigdf['gt_ix'].tolist(), graph=None, method_name="ground_truth")
        pred_ix_nc = NodeClustering(communities=cur_model_assigdf['pred_ix'].tolist(), graph=None, method_name="prediction")

        print("Computing ONMI LFK (can take ~20 minutes)")
        onmi_lfk = evaluation.overlapping_normalized_mutual_information_LFK(gt_nc, pred_nc)
        print("Computing ONMI MGH")
        onmi_mgh = evaluation.overlapping_normalized_mutual_information_MGH(gt_nc, pred_nc)
        print("Computing Omega")
        omega = evaluation.omega(gt_ix_nc, pred_ix_nc)
        #print("Computing ARI")
        #ari = evaluation.adjusted_rand_index(gt_ix_nc, pred_ix_nc)
        print("Computing F1")
        f1 = evaluation.f1(gt_ix_nc, pred_ix_nc)
        print("Computing Overlap Quality")
        overlap_q = evaluation.overlap_quality(gt_ix_nc, pred_ix_nc)
        print("Computing Variation of Information")
        voi = evaluation.variation_of_information(gt_ix_nc, pred_ix_nc)
        #print("Computing ECS")
        #ecs = evaluation.ecs(gt_ix_nc, pred_ix_nc)
        print("Computing Classification Error")
        classification_err = evaluation.classification_error(gt_ix_nc, pred_ix_nc)
        results_d = {
            **results_d,
            'onmi_lfk': onmi_lfk.score,
            'onmi_mgh': onmi_mgh.score,
            'omega': omega.score,
            #'ari': ari.score,
            'f1': f1.score,
            'overlap_quality': overlap_q.score,
            'variation_of_information': voi.score,
            #'ecs': ecs.score,
            'classification_error': classification_err.score,
        }

    results_allmodels.append(results_d)
results_allmodels_df = pd.DataFrame(results_allmodels).set_index('name')

print(f"Result dataframe:\n{results_allmodels_df}")

clustering_results_output_path = Path(output_dir / "clustering_scores.csv")

print(f"Saving output to: {clustering_results_output_path }")
results_allmodels_df.to_csv(clustering_results_output_path, index=False)