import math
import os
from typing import List, Optional, Dict, Literal
from pydantic import BaseModel
from transformers import BertModel, BertTokenizerFast
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, silhouette_score
from sklearn.linear_model import LinearRegression
from wordcloud import WordCloud
from tqdm import tqdm
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from itertools import zip_longest

class Assessment(BaseModel):
    id: Optional[int] = None
    revision_id: int
    reference_id: int
    type: Literal["semantic-similarity"]

def get_labse_model(cache_path="model_cache"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        print("Trying to load model from cache...")
        semsim_model = BertModel.from_pretrained(
            "setu4993/LaBSE", cache_dir=cache_path
        ).to(device).eval()
    except OSError as e:
        print(e)
        print("Downloading model instead of using cache...")
        semsim_model = BertModel.from_pretrained(
            "setu4993/LaBSE", cache_dir=cache_path, force_download=True
        ).to(device).eval()
    print("Semantic model initialized...")

    try:
        semsim_tokenizer = BertTokenizerFast.from_pretrained(
            "setu4993/LaBSE", cache_dir=cache_path
        )
    except OSError as e:
        print(e)
        print("Downloading tokenizer instead of using cache...")
        semsim_tokenizer = BertTokenizerFast.from_pretrained(
            "setu4993/LaBSE", cache_dir=cache_path, force_download=True
        )
    print("Tokenizer initialized...")

    return semsim_model, semsim_tokenizer, device

def get_sim_scores(
    rev_sents_output: List[str],
    ref_sents_output: List[str],
    semsim_model=None,
    semsim_tokenizer=None,
    device="cpu"
):
    if semsim_model is None or semsim_tokenizer is None:
        semsim_model, semsim_tokenizer, device = get_labse_model()

    print(f"rev_sents_output: {rev_sents_output[:5]}")
    print(f"ref_sents_output: {ref_sents_output[:5]}")

    rev_sents_input = semsim_tokenizer(
        rev_sents_output, return_tensors="pt", padding=True, truncation=True
    ).to(device)
    ref_sents_input = semsim_tokenizer(
        ref_sents_output, return_tensors="pt", padding=True, truncation=True
    ).to(device)

    with torch.no_grad():
        rev_sents_output = semsim_model(**rev_sents_input)
        ref_sents_output = semsim_model(**ref_sents_input)

    rev_sents_embedding = rev_sents_output.pooler_output
    ref_sents_embedding = ref_sents_output.pooler_output

    sim_scores = torch.nn.functional.cosine_similarity(
        rev_sents_embedding, ref_sents_embedding, dim=1
    ).tolist()

    return sim_scores, rev_sents_embedding

def descriptive_statistics(sim_scores):
    scores_array = np.array(sim_scores)
    print("Mean similarity:", np.mean(scores_array))
    print("Median similarity:", np.median(scores_array))
    print("Standard Deviation of similarity scores:", np.std(scores_array))
    plt.hist(scores_array, bins=20, alpha=0.75)
    plt.title('Distribution of Similarity Scores')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.show()

def cluster_verses_embeddings(embeddings):
    optimal_k = 2
    optimal_silhouette = -1
    for k in range(2, 10):  # Assuming a range for possible K values
        kmeans = KMeans(n_clusters=k, random_state=0).fit(embeddings)
        silhouette_avg = silhouette_score(embeddings, kmeans.labels_)
        print(f"Silhouette Score for k={k}: {silhouette_avg}")
        if silhouette_avg > optimal_silhouette:
            optimal_k = k
            optimal_silhouette = silhouette_avg
    # Final model with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=0).fit(embeddings)
    print(f"Optimal number of clusters: {optimal_k}")
    return kmeans.labels_

def analyze_correlation(data_frame: pd.DataFrame, column1: str, column2: str):
    correlation = data_frame[column1].corr(data_frame[column2])
    print(f"Correlation between {column1} and {column2}: {correlation}")
    plt.scatter(data_frame[column1], data_frame[column2])
    plt.title(f"Correlation between {column1} and {column2}")
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.show()

def plot_time_series(sim_scores: List[float]):
    plt.figure(figsize=(10, 5))
    plt.plot(sim_scores, label='Similarity Score')
    plt.title('Trend of Similarity Scores Over Verses')
    plt.xlabel('Verse Index')
    plt.ylabel('Similarity Score')
    plt.legend()
    plt.show()

def analyze_extreme_cases(revision_sentences: List[str], reference_sentences: List[str], sim_scores: List[float], num_cases: int = 5):
    sorted_indices = np.argsort(sim_scores)
    print("Lowest similarity verses:")
    for i in sorted_indices[:num_cases]:
        print(f"Revision: {revision_sentences[i]}")
        print(f"Reference: {reference_sentences[i]}")
        print(f"Score: {sim_scores[i]}\n")
    
    print("Highest similarity verses:")
    for i in sorted_indices[-num_cases:]:
        print(f"Revision: {revision_sentences[i]}")
        print(f"Reference: {reference_sentences[i]}")
        print(f"Score: {sim_scores[i]}\n")

def characterize_clusters(sentences: List[str], labels: List[int]):
    stop_words = set(stopwords.words('english'))
    cluster_contents = {i: [] for i in set(labels)}
    for sentence, label in zip(sentences, labels):
        words = word_tokenize(sentence)
        filtered_words = [word for word in words if word not in stop_words and word.isalnum()]
        cluster_contents[label].extend(filtered_words)
    
    for label, words in cluster_contents.items():
        word_freq = Counter(words)
        print(f"Most common words in cluster {label}: {word_freq.most_common(10)}")

def save_sim_scores_to_file(sim_scores: List[float], file_path: str):
    print("Saving the similarity scores to file")
    with open(file_path, 'w') as file:
        for score in sim_scores:
            file.write(f"{score}\n")

def save_results_to_file(results: List[dict], file_path: str):
    with open(file_path, 'w') as file:
        for result in results:
            file.write(f"{result}\n")

def get_text(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    encodings = ['utf-8', 'latin-1', 'ascii', 'utf-16']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
            return content
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to read the file with any of the encodings: {encodings}")

def replace_keyword_in_file(file_path: str, keyword: str = "<range>", replacement: str = " "):
    if not os.path.exists(file_path):
        print(f"File {file_path} not found. Skipping replacement.")
        return
    
    print("Replacing the keyword in the file")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        modified_lines = [line.replace(keyword, replacement) for line in lines]
        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(modified_lines)
        print(f"Replaced '{keyword}' with '{replacement}' in {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def get_line_numbers_from_vref(file_path: str) -> List[int]:
    print("Getting the line numbers from vref file")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        line_numbers = []
        for line in lines:
            try:
                line_number = int(line.strip().split()[-1])
                line_numbers.append(line_number)
            except ValueError:
                line_numbers.append(-1)
        return line_numbers
    except Exception as e:
        print(f"An error occurred while reading vref file: {e}")
        return []

def replace_lines_with_blank(file_path: str, line_numbers: List[int]):
    if not os.path.exists(file_path):
        print(f"File {file_path} not found. Skipping line replacement.")
        return

    print("Starting to replace the lines")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        for line_number in line_numbers:
            if line_number == -1:
                lines.append(' \n')
            elif 0 <= line_number - 1 < len(lines):
                lines[line_number - 1] = ' \n'
        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(lines)
        print(f"Replaced specified lines with blanks in {file_path}")
    except Exception as e:
        print(f"An error occurred while replacing lines: {e}")

def merge_files(file1: str, file2: str, output_file: str):
    print("Merging files")
    try:
        with open(file1, 'r') as f1, open(file2, 'r') as f2, open(output_file, 'w') as outfile:
            for line1, line2 in zip_longest(f1, f2, fillvalue=''):
                outfile.write(f"{line1.strip()}\t{line2.strip()}\n")
        print(f"Merged files into {output_file}")
    except Exception as e:
        print(f"An error occurred while merging files: {e}")

def removeverse(vref_file: str, merged_file: str):
    print("Removing verses from merged file")
    try:
        with open(vref_file, 'r') as vref, open(merged_file, 'r') as merged, open('final_output.txt', 'w') as final:
            vref_lines = set(vref.read().splitlines())
            for line in merged:
                if line.split('\t')[0] not in vref_lines:
                    final.write(line)
        print(f"Verses removed. Final output written to 'final_output.txt'")
    except Exception as e:
        print(f"An error occurred while removing verses: {e}")

def assess():
    assessment = {
        "revision_id": 1, 
        "reference_id": 1, 
        "type": "semantic-similarity"
    }

    if isinstance(assessment, dict):
        assessment = Assessment(**assessment)

    # Absolute paths to text files on the system
    revision_file_path = r"./newdata/tur-turev.txt"
    reference_file_path = r"./newdata/eng-eng-kjv2006.txt"
    
    replace_keyword_in_file(revision_file_path)
    replace_keyword_in_file(reference_file_path)

    line_numbers = get_line_numbers_from_vref("references/vref_file.txt")
    replace_lines_with_blank(revision_file_path, line_numbers)
    replace_lines_with_blank(reference_file_path, line_numbers)

    try:
        revision_text = get_text(revision_file_path)
        reference_text = get_text(reference_file_path)
    except FileNotFoundError as e:
        print(e)
        return None

    revision_lines = revision_text.split('\n')
    reference_lines = reference_text.split('\n')

    revision_df = pd.DataFrame(revision_lines, columns=["revision"])
    reference_df = pd.DataFrame(reference_lines, columns=["reference"])

    merged_df = pd.concat([revision_df, reference_df], axis=1)
    print(merged_df.size)

    batch_size = 256
    rev_sents = merged_df["revision"].to_list()
    ref_sents = merged_df["reference"].to_list()
    vrefs = merged_df.index.to_list()
    assessment_id = [assessment.id] * len(vrefs)
    rev_sents_batched = [
        rev_sents[i : i + batch_size] for i in range(0, len(rev_sents), batch_size)
    ]
    ref_sents_batched = [
        ref_sents[i : i + batch_size] for i in range(0, len(ref_sents), batch_size)
    ]
    
    semsim_model, semsim_tokenizer, device = get_labse_model()
    sim_scores = []
    
    for i, (rev_batch, ref_batch) in enumerate(tqdm(zip(rev_sents_batched, ref_sents_batched), total=len(rev_sents_batched))):
        rev_batch = [sent if sent else " " for sent in rev_batch]
        ref_batch = [sent if sent else " " for sent in ref_batch]
        
        try:
            batch_scores, rev_sents_embedding = get_sim_scores(rev_batch, ref_batch, semsim_model, semsim_tokenizer, device)
            sim_scores.extend(batch_scores)
        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            print(f"rev_batch: {rev_batch}")
            print(f"ref_batch: {ref_batch}")
            raise e

    results = [
        {
            "vref": vrefs[j],
            "score": sim_scores[j] if not math.isnan(sim_scores[j]) else 0,
        }
        for j in range(len(vrefs))
    ]

    save_sim_scores_to_file(sim_scores, "references/sim_scores.txt")
    merge_files('references/vref.txt', 'references/sim_scores.txt', 'references/merged_results.txt')
    removeverse("references/vref_file.txt", "references/merged_results.txt")

    return {"results": results}

# Run the assessment and save results to a file
output = assess()
if output:
    save_results_to_file(output["results"], "./LABsE/resultsLABsE.txt")

# Perform additional analyses if output is not None
if output:
    descriptive_statistics([result['score'] for result in output["results"]])
    plot_time_series([result['score'] for result in output["results"]])
    analyze_extreme_cases(
        [result['revision'] for result in output["results"]],
        [result['reference'] for result in output["results"]],
        [result['score'] for result in output["results"]]
    )
    labels = cluster_verses_embeddings(np.array([result['score'] for result in output["results"]]).reshape(-1, 1))
    characterize_clusters([result['revision'] for result in output["results"]], labels)