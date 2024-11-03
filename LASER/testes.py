import os
import pandas as pd
from pydantic import BaseModel
from typing import List, Optional, Literal
import torch
import numpy as np
from laserembeddings import Laser  # type: ignore
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from itertools import zip_longest
from collections import Counter
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LinearRegression
from transformers import pipeline
from scipy.stats import ttest_ind

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1

# Initialize NER pipeline with a faster model and enable batching
ner_pipeline = pipeline(
    "ner",
    model="elastic/distilbert-base-uncased-finetuned-conll03-english",
    tokenizer="elastic/distilbert-base-uncased-finetuned-conll03-english",
    grouped_entities=True,
    device=device,
)

class Assessment(BaseModel):
    id: Optional[int] = None
    revision_id: int
    reference_id: int
    type: Literal["semantic-similarity"]

def get_text(file_path: str):
    print("Reading the file")
    encodings = ['utf-8', 'latin-1', 'ascii', 'utf-16']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
            return content
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to read the file with any of the encodings: {encodings}")

def get_sim_scores_and_embeddings(rev_sents_output: List[str], ref_sents_output: List[str]):
    print("Getting the similarity scores and embeddings")
    laser = Laser()
    rev_sents_embedding = laser.embed_sentences(rev_sents_output, lang='en')
    ref_sents_embedding = laser.embed_sentences(ref_sents_output, lang='en')
    sim_scores = torch.nn.functional.cosine_similarity(
        torch.tensor(rev_sents_embedding),
        torch.tensor(ref_sents_embedding),
        dim=1
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

def save_sim_scores_to_file(sim_scores_with_names, sim_scores_no_names, names_presence, file_path):
    print("Saving the similarity scores to file")
    with open(file_path, 'w') as file:
        for score_with_name, score_no_name, has_name in zip(sim_scores_with_names, sim_scores_no_names, names_presence):
            file.write(f"{score_with_name}\t{score_no_name}\t{has_name}\n")

def replace_keyword_in_file(file_path: str, keyword: str = "<range>", replacement: str = " "):
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

def get_line_numbers_from_vref(file_path: str):
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
        print(f"Replaced specified lines with blank spaces in {file_path}")
    except Exception as e:
        print(f"An error occurred while processing the file {file_path}: {e}")

def merge_files(file1_path, file2_path, output_path):
    print("Merging the files")
    with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2, open(output_path, 'w') as output:
        # Use zip_longest to handle files of different lengths
        for line1, line2 in zip_longest(file1, file2, fillvalue=''):
            # Strip newline characters and combine the lines
            merged_line = line1.strip() + ' ' + line2.strip() + '\n'
            output.write(merged_line)

def removeverse(reference_path, result_path):
    print("Removing the verse")
    with open(reference_path, 'r') as vref_file:
        vref_lines = vref_file.readlines()
    with open(result_path, 'r') as merged_file:
        merged_lines = merged_file.readlines()
    # Extract references to look for
    vref_references = [line.strip() for line in vref_lines]
    # Open the merged results file to write the updated content
    with open('references/merged_results.txt', 'w') as merged_file_updated:
        for line in merged_lines:
            if any(ref in line for ref in vref_references):
                merged_file_updated.write('\n')  # Write a blank line
            else:
                merged_file_updated.write(line)  # Write the original line
    print("The references have been replaced with blank lines in the updated file.")

def analyze_correlation(data_frame, column1, column2):
    correlation = data_frame[column1].corr(data_frame[column2])
    print(f"Correlation between {column1} and {column2}: {correlation}")
    plt.scatter(data_frame[column1], data_frame[column2])
    plt.title(f"Correlation between {column1} and {column2}")
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.show()

def plot_time_series(sim_scores):
    plt.figure(figsize=(10, 5))
    plt.plot(sim_scores, label='Similarity Score')
    plt.title('Trend of Similarity Scores Over Verses')
    plt.xlabel('Verse Index')
    plt.ylabel('Similarity Score')
    plt.legend()
    plt.show()

def analyze_extreme_cases(revision_sentences, reference_sentences, sim_scores, num_cases=5):
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

def characterize_clusters(sentences, labels):
    stop_words = set(stopwords.words('english'))
    cluster_contents = {i: [] for i in set(labels)}
    for sentence, label in zip(sentences, labels):
        words = word_tokenize(sentence)
        filtered_words = [word.lower() for word in words if word.lower() not in stop_words and word.isalnum()]
        cluster_contents[label].extend(filtered_words)

    for label, words in cluster_contents.items():
        word_freq = Counter(words)
        print(f"Most common words in cluster {label}: {word_freq.most_common(10)}")

def regression_analysis(features, sim_scores):
    model = LinearRegression()
    model.fit(features, sim_scores)
    predictions = model.predict(features)
    plt.scatter(sim_scores, predictions)
    plt.xlabel('Actual Scores')
    plt.ylabel('Predicted Scores')
    plt.title('Regression Analysis Results')
    plt.plot([min(sim_scores), max(sim_scores)], [min(predictions), max(predictions)], color='red')  # Line of best fit
    plt.show()

def detect_names_in_sentences(sentences):
    print("Detecting names in sentences using batch processing with transformers")
    names_presence = []
    batch_size = 32  # Adjust batch size based on your system's memory
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i+batch_size]
        ner_results_batch = ner_pipeline(batch_sentences)
        # If the batch returns a list of results per sentence
        for ner_results in ner_results_batch:
            has_name = any(ent['entity_group'] == 'PER' for ent in ner_results)
            names_presence.append(has_name)
    return names_presence

def remove_names_from_sentences(sentences):
    print("Removing names from sentences")
    modified_sentences = []
    batch_size = 32  # Adjust based on your system
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i+batch_size]
        ner_results_batch = ner_pipeline(batch_sentences)
        for sentence, ner_results in zip(batch_sentences, ner_results_batch):
            # Ensure ner_results is a list
            if isinstance(ner_results, dict):
                ner_results = [ner_results]
            # Sort entities in reverse order of 'start'
            entities = [ent for ent in ner_results if ent['entity_group'] == 'PER']
            entities = sorted(entities, key=lambda x: x['start'], reverse=True)
            for ent in entities:
                start = ent['start']
                end = ent['end']
                sentence = sentence[:start] + '' + sentence[end:]
            modified_sentences.append(sentence)
    return modified_sentences

def compare_similarity_with_names(sim_scores, names_presence):
    print("Comparing similarity scores based on the presence of names")
    scores_with_names = [score for score, has_name in zip(sim_scores, names_presence) if has_name]
    scores_without_names = [score for score, has_name in zip(sim_scores, names_presence) if not has_name]
    t_stat, p_value = ttest_ind(scores_with_names, scores_without_names, equal_var=False)
    print(f"\nSimilarity Scores Analysis Based on Presence of Names:")
    print(f"Average similarity (with names): {np.mean(scores_with_names):.4f}")
    print(f"Average similarity (without names): {np.mean(scores_without_names):.4f}")
    print(f"T-test p-value: {p_value:.4f}\n")

def main():
    vref_file_path = "references/vref_file.txt"
    revision_file_path = "data/aai-aai.txt"
    reference_file_path = "data/aak-aak.txt"

    replace_keyword_in_file(revision_file_path)
    replace_keyword_in_file(reference_file_path)

    line_numbers = get_line_numbers_from_vref(vref_file_path)

    replace_lines_with_blank(revision_file_path, line_numbers)
    replace_lines_with_blank(reference_file_path, line_numbers)

    revision_text = get_text(revision_file_path)
    reference_text = get_text(reference_file_path)

    revision_sentences = revision_text.split('\n')
    reference_sentences = reference_text.split('\n')

    # Ensure both lists have the same length
    min_length = min(len(revision_sentences), len(reference_sentences))
    revision_sentences = revision_sentences[:min_length]
    reference_sentences = reference_sentences[:min_length]

    # Detect names in revision sentences
    names_presence = detect_names_in_sentences(revision_sentences)

    # Calculate similarity scores with names
    sim_scores_with_names, embeddings = get_sim_scores_and_embeddings(revision_sentences, reference_sentences)

    # Remove names from sentences
    modified_revision_sentences = remove_names_from_sentences(revision_sentences)
    modified_reference_sentences = remove_names_from_sentences(reference_sentences)

    # Calculate similarity scores without names
    sim_scores_no_names, _ = get_sim_scores_and_embeddings(modified_revision_sentences, modified_reference_sentences)

    # Save similarity scores and names presence to file
    save_sim_scores_to_file(sim_scores_with_names, sim_scores_no_names, names_presence, "references/sim_scores.txt")

    # Perform analysis with original similarity scores
    descriptive_statistics(sim_scores_with_names)

    labels = cluster_verses_embeddings(embeddings)
    print("Cluster labels for verses:", labels)

    plot_time_series(sim_scores_with_names)
    analyze_extreme_cases(revision_sentences, reference_sentences, sim_scores_with_names)
    characterize_clusters(revision_sentences, labels)

    regression_analysis(embeddings, sim_scores_with_names)

    merge_files('references/vref.txt', 'references/sim_scores.txt', 'references/merged_results.txt')
    merge_data_path = 'references/merged_results.txt'
    removeverse(vref_file_path, merge_data_path)

    # Compare similarity scores based on presence of names
    compare_similarity_with_names(sim_scores_with_names, names_presence)

    print("The process has been completed")

if __name__ == "__main__":
    main()
