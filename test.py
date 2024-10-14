# def merge_files(file1_path, file2_path, output_path):
#     with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2, open(output_path, 'w') as output:
#         # Use zip_longest to handle files of different lengths
#         from itertools import zip_longest
        
#         # Iterate over both files simultaneously
#         for line1, line2 in zip_longest(file1, file2, fillvalue=''):
#             # Strip newline characters and combine the lines
#             merged_line = line1.strip() +' '+ line2.strip() + '\n'
#             output.write(merged_line)

# def removeverse(refernce_path, result_path):
#     with open(refernce_path, 'r') as vref_file:
#         vref_lines = vref_file.readlines()

#     with open(result_path, 'r') as merged_file:
#         merged_lines = merged_file.readlines()

#     # Extract references to look for
#     vref_references = [line.strip() for line in vref_lines]

#     # Open the merged results file to write the updated content
#     with open('data/merged_results.txt', 'w') as merged_file_updated:
#         for line in merged_lines:
#             if any(ref in line for ref in vref_references):
#                 merged_file_updated.write('\n')  # Write a blank line
#             else:
#                 merged_file_updated.write(line)  # Write the original line

#     print("The references have been replaced with blank lines in the updated file.")




# def main():
#     revision_file_path, reference_file_path = select_files()
    
#     revision_filename = os.path.basename(revision_file_path).split('.')[0]
#     reference_filename = os.path.basename(reference_file_path).split('.')[0]

#     sim_scores_filename = f"{revision_filename}-{reference_filename}-sim-scores.txt"
#     merged_results_filename = f"{revision_filename}-{reference_filename}-merged_results.txt"
    
#     replace_keyword_in_file(revision_file_path)
#     replace_keyword_in_file(reference_file_path)
    
#     line_numbers = get_line_numbers_from_vref("references/vref_file.txt")
    
#     replace_lines_with_blank(revision_file_path, line_numbers)
#     replace_lines_with_blank(reference_file_path, line_numbers)
    
#     revision_text = get_text(revision_file_path)
#     reference_text = get_text(reference_file_path)
    
#     revision_sentences = revision_text.split('\n')
#     reference_sentences = reference_text.split('\n')
    
#     sim_scores, embeddings = get_sim_scores_and_embeddings(revision_sentences, reference_sentences)
    
#     save_sim_scores_to_file(sim_scores, f"references/{sim_scores_filename}")
    
#     descriptive_statistics(sim_scores)
    
#     labels = cluster_verses_embeddings(embeddings)
#     print("Cluster labels for verses:", labels)
    
#     plot_time_series(sim_scores)
#     analyze_extreme_cases(revision_sentences, reference_sentences, sim_scores)
#     characterize_clusters(revision_sentences, labels)

#     regression_analysis(embeddings, sim_scores)
    
#     merge_files('references/vref.txt', f"references/{sim_scores_filename}", f"references/{merged_results_filename}")
#     merge_data_path = f"references/{merged_results_filename}"
#     removeverse("references/vref_file.txt", merge_data_path)
    
#     print("The process has been completed")




import pandas as pd
import torch
from laserembeddings import Laser
import matplotlib.pyplot as plt
from transformers import pipeline,BertModel, BertTokenizerFast
from typing import List, Optional, Literal
from pydantic import BaseModel


device = 0 if torch.cuda.is_available() else -1
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

def read_file_to_dataframe(file_path: str, column_name: str):
    print(f"Reading file {file_path} into DataFrame")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
        df = pd.DataFrame(lines, columns=[column_name])
        return df
    except UnicodeDecodeError as e:
        print(f"Error reading {file_path}: {e}")
        raise

def replace_keyword_in_dataframe(df: pd.DataFrame, column_name: str, keyword: str = "<range>", replacement: str = " "):
    print(f"Replacing '{keyword}' with '{replacement}' in DataFrame column '{column_name}'")
    df[column_name] = df[column_name].str.replace(keyword, replacement, regex=False)
    return df

def replace_verses_with_blank_in_dataframe(df: pd.DataFrame, vref_df: pd.DataFrame, text_column: str, vref_column: str):
    print("Replacing specified verses with blank spaces in DataFrame")
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()
    vref_list = vref_df[vref_column].tolist()

    # Assuming the verses are line numbers
    for verse in vref_list:
        try:
            # Adjusting for zero-based index
            verse_index = int(verse) - 1
            if 0 <= verse_index < len(df):
                df.at[verse_index, text_column] = ' '
        except ValueError:
            print(f"Invalid verse reference: {verse}")
            continue
    return df



def get_labse_model(cache_path="model_cache"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    semsim_model = BertModel.from_pretrained(
        "setu4993/LaBSE", cache_dir=cache_path
    ).to(device).eval()
    
    semsim_tokenizer = BertTokenizerFast.from_pretrained(
        "setu4993/LaBSE", cache_dir=cache_path
    )

    return semsim_model, semsim_tokenizer, device

def get_sim_scores_and_embeddings_from_sentences(rev_sents_output: List[str], ref_sents_output: List[str]):
    print("Calculating similarity scores and embeddings from sentences")
    
    semsim_model, semsim_tokenizer, device = get_labse_model()

    def embed_sentences(sentences):
        inputs = semsim_tokenizer(sentences, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            embeddings = semsim_model(**inputs).pooler_output
        return embeddings

    rev_sents_embedding = embed_sentences(rev_sents_output)
    ref_sents_embedding = embed_sentences(ref_sents_output)

    sim_scores = torch.nn.functional.cosine_similarity(rev_sents_embedding, ref_sents_embedding, dim=1).cpu().tolist()
    
    return sim_scores, rev_sents_embedding

def merge_dataframes(df_list: List[pd.DataFrame]):
    print("Merging DataFrames")
    merged_df = pd.concat(df_list, axis=1)
    return merged_df

def append_vref_txt(merged_df: pd.DataFrame, vref_txt_path: str):
    print("Appending vref.txt content to DataFrame")
    vref_txt_df = read_file_to_dataframe(vref_txt_path, 'vref_mapped')
    merged_df_with_vref = pd.concat([merged_df.reset_index(drop=True), vref_txt_df.reset_index(drop=True)], axis=1)
    return merged_df_with_vref

def analyze_correlation_in_dataframe(df: pd.DataFrame, column1: str, column2: str):
    correlation = df[column1].corr(df[column2])
    print(f"Correlation between {column1} and {column2}: {correlation}")
    plt.scatter(df[column1], df[column2])
    plt.title(f"Correlation between {column1} and {column2}")
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.show()

def main():
    vref_file_path = "references/vref_file.txt"
    revision_file_path = "newdata/tur-turev.txt"
    reference_file_path = "newdata/eng-eng-kjv2006.txt"
    vref_txt_path = "references/vref.txt"

    # Read files into DataFrames
    revision_df = read_file_to_dataframe(revision_file_path, 'revision_sentence')
    reference_df = read_file_to_dataframe(reference_file_path, 'reference_sentence')
    vref_df = read_file_to_dataframe(vref_file_path, 'verse_reference')

    # Replace keywords in DataFrames
    revision_df = replace_keyword_in_dataframe(revision_df, 'revision_sentence')
    reference_df = replace_keyword_in_dataframe(reference_df, 'reference_sentence')

    # Replace specified verses with blank spaces
    revision_df = replace_verses_with_blank_in_dataframe(revision_df, vref_df, 'revision_sentence', 'verse_reference')
    reference_df = replace_verses_with_blank_in_dataframe(reference_df, vref_df, 'reference_sentence', 'verse_reference')

    # Ensure revision and reference DataFrames have the same length
    min_length = min(len(revision_df), len(reference_df))
    revision_df = revision_df.iloc[:min_length].reset_index(drop=True)
    reference_df = reference_df.iloc[:min_length].reset_index(drop=True)
    # Note: Do not adjust vref_df length

    # Identify non-empty rows where both sentences are not empty
    non_empty_rows = (revision_df['revision_sentence'].str.strip() != '') & (reference_df['reference_sentence'].str.strip() != '')

    # Initialize sim_scores with blanks
    sim_scores = ['' for _ in range(len(revision_df))]

    # Extract sentences where both are non-empty
    rev_sentences_non_empty = revision_df.loc[non_empty_rows, 'revision_sentence'].tolist()
    ref_sentences_non_empty = reference_df.loc[non_empty_rows, 'reference_sentence'].tolist()

    # Compute sim_scores for these sentences
    if rev_sentences_non_empty and ref_sentences_non_empty:
        computed_sim_scores, _ = get_sim_scores_and_embeddings_from_sentences(rev_sentences_non_empty, ref_sentences_non_empty)
        # Assign computed sim_scores back to the appropriate positions
        sim_scores_indices = revision_df.loc[non_empty_rows].index.tolist()
        for idx, score in zip(sim_scores_indices, computed_sim_scores):
            sim_scores[idx] = score

    # Create DataFrame for similarity scores
    sim_scores_df = pd.DataFrame({'sim_scores': sim_scores})

    # Merge DataFrames
    merged_df = merge_dataframes([vref_df.reset_index(drop=True), revision_df, reference_df, sim_scores_df])

    # Append vref.txt content
    merged_df = append_vref_txt(merged_df, vref_txt_path)

    # Save the final DataFrame to both a text file and a CSV file
    merged_df.to_csv('references/merged_results.txt', index=False, sep='\t')
    merged_df.to_csv('references/merged_results.csv', index=False)

    print("The process has been completed")

if _name_ == "_main_":
    main()