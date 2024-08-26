import os
import pandas as pd
from pydantic import BaseModel
from typing import List, Optional, Literal
import torch
from laserembeddings import Laser # type: ignore

class Assessment(BaseModel):
    id: Optional[int] = None
    revision_id: int
    reference_id: int
    type: Literal["semantic-similarity"]

def get_text(file_path: str):
    encodings = ['utf-8', 'latin-1', 'ascii', 'utf-16']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
            return content
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to read the file with any of the encodings: {encodings}")

def get_sim_scores(rev_sents_output: List[str], ref_sents_output: List[str]):
    laser = Laser()
    rev_sents_embedding = laser.embed_sentences(rev_sents_output, lang='en')
    ref_sents_embedding = laser.embed_sentences(ref_sents_output, lang='en')
    sim_scores = torch.nn.functional.cosine_similarity(
        torch.tensor(rev_sents_embedding),
        torch.tensor(ref_sents_embedding),
        dim=1
    ).tolist()
    return sim_scores

def save_sim_scores_to_file(sim_scores, file_path):
    with open(file_path, 'w') as file:
        for score in sim_scores:
            file.write(f"{score}\n")

def replace_keyword_in_file(file_path: str, keyword: str = "<range>", replacement: str = " "):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        modified_lines = [line.replace(keyword, replacement) for line in lines]
        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(modified_lines)
        print(f"Replaced '{keyword}' with '{replacement}' in {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def assess():
    revision_file_path = "data/aai-aai.txt"
    reference_file_path = "data/aak-aak.txt"
    revision_text = get_text(revision_file_path)
    reference_text = get_text(reference_file_path)
    revision_sentences = revision_text.split('\n')
    reference_sentences = reference_text.split('\n')
    sim_scores = get_sim_scores(revision_sentences, reference_sentences)
    return sim_scores

def merge_files(file1, file2, merged_file):
    try:
        with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
            content1 = f1.readlines()
            content2 = f2.readlines()
        merged_content = content1 + content2
        with open(merged_file, 'w', encoding='utf-8') as mf:
            mf.writelines(merged_content)
        print(f"Files {file1} and {file2} merged into {merged_file}")
    except Exception as e:
        print(f"Error while merging files: {e}")

def main():
    vref_file_path = "data/vref_file.txt"
    revision_file_path = "data/aai-aai.txt"
    reference_file_path = "data/aak-aak.txt"
    sim_scores_file = "sim_scores.txt"
    merged_results_file = "data/merged_results.txt"
    
    replace_keyword_in_file(revision_file_path)
    replace_keyword_in_file(reference_file_path)
    
    sim_scores = assess()
    save_sim_scores_to_file(sim_scores, sim_scores_file)
    
    merge_files(sim_scores_file, vref_file_path, merged_results_file)

if __name__ == "__main__":
    main()
