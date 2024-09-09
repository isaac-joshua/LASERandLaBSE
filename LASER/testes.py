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
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LinearRegression
import tkinter as tk
from tkinter import filedialog
from laser import plot_time_series,analyze_extreme_cases,characterize_clusters,regression_analysis

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
    
    # Convert NumPy arrays to PyTorch tensors
    rev_sents_tensor = torch.tensor(rev_sents_embedding)
    ref_sents_tensor = torch.tensor(ref_sents_embedding)

    # Calculate similarity scores
    sim_scores = torch.nn.functional.cosine_similarity(rev_sents_tensor, ref_sents_tensor, dim=1).tolist()

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

def save_sim_scores_to_file(sim_scores, file_path):
    print("Saving the similarity scores to file")
    with open(file_path, 'w') as file:
        for score in sim_scores:
            file.write(f"{score}\n")

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
        for line1, line2 in zip_longest(file1, file2, fillvalue=''):
            merged_line = line1.strip() + ' ' + line2.strip() + '\n'
            output.write(merged_line)
    print(f"Files {file1_path} and {file2_path} have been merged into {output_path}.")

def removeverse(reference_path, result_path):
    print("Removing the verse")
    
    with open(reference_path, 'r') as vref_file:
        vref_lines = vref_file.readlines()
    
    with open(result_path, 'r') as merged_file:
        merged_lines = merged_file.readlines()
    
    # Extract references to look for
    vref_references = [line.strip() for line in vref_lines]
    
    # Construct the updated file path based on input filenames
    updated_file_path = result_path.replace('merged_results', 'merged_results_removedverse')
    
    # Open the merged results file to write the updated content
    with open(updated_file_path, 'w') as merged_file_updated:
        for line in merged_lines:
            # If any reference is found in the line, replace the line with a blank
            if any(ref in line for ref in vref_references):
                merged_file_updated.write('\n')  # Write a blank line
            else:
                merged_file_updated.write(line)  # Write the original line
    
    print(f"The references have been replaced with blank lines in {updated_file_path}.")

def select_files():
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Select the revision file
    revision_file_path = filedialog.askopenfilename(title="Select the Revision File",
                                                    filetypes=[("Text Files", "*.txt")])
    
    # Select the reference file
    reference_file_path = filedialog.askopenfilename(title="Select the Reference File",
                                                     filetypes=[("Text Files", "*.txt")])
    
    return revision_file_path, reference_file_path

def main():
    revision_file_path, reference_file_path = select_files()
    
    revision_filename = os.path.basename(revision_file_path).split('-')[0]
    reference_filename = os.path.basename(reference_file_path).split('-')[0]

    sim_scores_filename = f"{revision_filename}-{reference_filename}-sim-scores.txt"
    merged_results_filename = f"{revision_filename}-{reference_filename}-merged_results.txt"
    
    replace_keyword_in_file(revision_file_path)
    replace_keyword_in_file(reference_file_path)
    
    line_numbers = get_line_numbers_from_vref("references/vref_file.txt")
    
    replace_lines_with_blank(revision_file_path, line_numbers)
    replace_lines_with_blank(reference_file_path, line_numbers)
    
    revision_text = get_text(revision_file_path)
    reference_text = get_text(reference_file_path)
    
    revision_sentences = revision_text.split('\n')
    reference_sentences = reference_text.split('\n')
    
    sim_scores, embeddings = get_sim_scores_and_embeddings(revision_sentences, reference_sentences)
    
    save_sim_scores_to_file(sim_scores, f"references/{sim_scores_filename}")
    
    # descriptive_statistics(sim_scores)
    # labels = cluster_verses_embeddings(embeddings)
    # print("Cluster labels for verses:", labels)
    # plot_time_series(sim_scores)
    # analyze_extreme_cases(revision_sentences, reference_sentences, sim_scores)
    # characterize_clusters(revision_sentences, labels)

    # regression_analysis(embeddings, sim_scores)
    
    merge_files('references/vref.txt', f"references/{sim_scores_filename}", f"references/{merged_results_filename}")
    merge_data_path = f"references/{merged_results_filename}"
    removeverse("references/vref_file.txt", merge_data_path)
    
    print("The process has been completed")

if __name__ == "__main__":
    main()
