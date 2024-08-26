import os
import pandas as pd
from pydantic import BaseModel
from typing import List, Optional, Literal
import torch
import numpy as np
from laserembeddings import Laser # type: ignore
# from test import  removeverse

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

def get_sim_scores(rev_sents_output: List[str],ref_sents_output: List[str],):
    print("Getting the similarity scores")
    laser = Laser()
    rev_sents_embedding = laser.embed_sentences(rev_sents_output, lang='en')
    ref_sents_embedding = laser.embed_sentences(ref_sents_output, lang='en')
    sim_scores = torch.nn.functional.cosine_similarity(
        torch.tensor(rev_sents_embedding),
        torch.tensor(ref_sents_embedding),
        dim=1
    ).tolist()
    return sim_scores

def assess(revision,reference):
    print("Assessing the similarity")
    revision_file_path = revision
    reference_file_path = reference

    revision_text = get_text(revision_file_path)
    reference_text = get_text(reference_file_path)

    revision_sentences = revision_text.split('\n')
    reference_sentences = reference_text.split('\n')

    sim_scores = get_sim_scores(revision_sentences, reference_sentences)

    return sim_scores

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
                print(line_numbers)
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
        from itertools import zip_longest
        # Iterate over both files simultaneously
        for line1, line2 in zip_longest(file1, file2, fillvalue=''):
            # Strip newline characters and combine the lines
            merged_line = line1.strip() +' '+ line2.strip() + '\n'
            output.write(merged_line)

def removeverse(refernce_path, result_path):
    print("Removing the verse")
    with open(refernce_path, 'r') as vref_file:
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

def main():
    vref_file_path = "references/vref_file.txt"
    revision_file_path = "data/aai-aai.txt"
    reference_file_path = "data/aak-aak.txt"
    
    replace_keyword_in_file(revision_file_path)
    replace_keyword_in_file(reference_file_path)
    
    line_numbers = get_line_numbers_from_vref(vref_file_path)
    
    replace_lines_with_blank(revision_file_path, line_numbers)
    replace_lines_with_blank(reference_file_path, line_numbers)
    
    sim_scores = assess(revision_file_path,reference_file_path)
    save_sim_scores_to_file(sim_scores, "references/sim_scores.txt")

    merge_files('references/vref.txt', 'references/sim_scores.txt', 'references/merged_results.txt')
    merge_data_path = 'references/merged_results.txt'
    removeverse(vref_file_path , merge_data_path)
    print("The process has been completed")
if __name__ == "__main__":
    main()