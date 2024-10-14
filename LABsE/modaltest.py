import pandas as pd
import torch
import modal
from transformers import BertModel, BertTokenizerFast
from typing import List, Optional, Literal
from pydantic import BaseModel

# Initialize Modal App
app = modal.App("labse-embedding")

# Modal function for embedding
@app.function(cpu=2, memory=8)  # Modify as needed
def get_labse_embeddings(rev_sents_output: List[str], ref_sents_output: List[str]) -> (List[float], List):
    print("Calculating similarity scores and embeddings in the remote worker")

    # Load model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    semsim_model = BertModel.from_pretrained("setu4993/LaBSE").to(device).eval()
    semsim_tokenizer = BertTokenizerFast.from_pretrained("setu4993/LaBSE")

    def embed_sentences(sentences):
        inputs = semsim_tokenizer(sentences, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            embeddings = semsim_model(**inputs).pooler_output
        return embeddings

    # Compute embeddings
    rev_sents_embedding = embed_sentences(rev_sents_output)
    ref_sents_embedding = embed_sentences(ref_sents_output)

    sim_scores = torch.nn.functional.cosine_similarity(rev_sents_embedding, ref_sents_embedding, dim=1).cpu().tolist()
    
    return sim_scores, rev_sents_embedding.cpu().numpy().tolist()

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
    df = df.copy()
    vref_list = vref_df[vref_column].tolist()

    for verse in vref_list:
        try:
            verse_index = int(verse) - 1
            if 0 <= verse_index < len(df):
                df.at[verse_index, text_column] = ' '
        except ValueError:
            print(f"Invalid verse reference: {verse}")
            continue
    return df

def merge_dataframes(df_list: List[pd.DataFrame]):
    print("Merging DataFrames")
    merged_df = pd.concat(df_list, axis=1)
    return merged_df

def append_vref_txt(merged_df: pd.DataFrame, vref_txt_path: str):
    print("Appending vref.txt content to DataFrame")
    vref_txt_df = read_file_to_dataframe(vref_txt_path, 'vref_mapped')
    merged_df_with_vref = pd.concat([merged_df.reset_index(drop=True), vref_txt_df.reset_index(drop=True)], axis=1)
    return merged_df_with_vref

@app.local_entrypoint()
def main():
    vref_file_path = "references/vref_file.txt"
    revision_file_path = "SmallData27/tur-turev.txt"
    reference_file_path = "SmallData27/eng-eng-kjv2006.txt"
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

    non_empty_rows = (revision_df['revision_sentence'].str.strip() != '') & (reference_df['reference_sentence'].str.strip() != '')

    sim_scores = ['' for _ in range(len(revision_df))]

    rev_sentences_non_empty = revision_df.loc[non_empty_rows, 'revision_sentence'].tolist()
    ref_sentences_non_empty = reference_df.loc[non_empty_rows, 'reference_sentence'].tolist()

    if rev_sentences_non_empty and ref_sentences_non_empty:
        # Execute on the Modal remote worker
        computed_sim_scores, _ = get_labse_embeddings.call(rev_sentences_non_empty, ref_sentences_non_empty)
        sim_scores_indices = revision_df.loc[non_empty_rows].index.tolist()
        for idx, score in zip(sim_scores_indices, computed_sim_scores):
            sim_scores[idx] = score

    sim_scores_df = pd.DataFrame({'sim_scores': sim_scores})

    merged_df = merge_dataframes([vref_df.reset_index(drop=True), revision_df, reference_df, sim_scores_df])

    merged_df = append_vref_txt(merged_df, vref_txt_path)

    merged_df.to_csv('references/merged_results.csv', index=False)

    print("The process has been completed")

if __name__ == "__main__":
    app.run()