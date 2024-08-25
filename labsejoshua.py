import math
import os
from typing import Literal, Optional, List
import pandas as pd
from pydantic import BaseModel
from transformers import BertTokenizerFast, BertModel
import torch
from tqdm import tqdm

class Assessment(BaseModel):
    id: Optional[int] = None
    revision_id: int
    reference_id: int
    type: Literal["semantic-similarity"]

def get_labse_model(cache_path="model_cache"):
    try:
        print("Trying to load model from cache...")
        semsim_model = BertModel.from_pretrained(
            "setu4993/LaBSE", cache_dir=cache_path
        ).eval()
    except OSError as e:
        print(e)
        print("Downloading model instead of using cache...")
        semsim_model = BertModel.from_pretrained(
            "setu4993/LaBSE", cache_dir=cache_path, force_download=True
        ).eval()
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

    return semsim_model, semsim_tokenizer

def get_sim_scores(
    rev_sents_output: List[str],
    ref_sents_output: List[str],
    semsim_model=None,
    semsim_tokenizer=None,
    device=torch.device("cpu")
):
    if semsim_model is None or semsim_tokenizer is None:
        semsim_model, semsim_tokenizer = get_labse_model()
    
    # Debugging: Print out the first few sentences
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

    sim_scores = torch.nn.CosineSimilarity(dim=1, eps=1e-6)(
        rev_sents_embedding, ref_sents_embedding
    ).tolist()

    return sim_scores

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

def save_sim_scores_to_file(sim_scores: List[float], file_path: str):
    with open(file_path, 'w') as file:
        for score in sim_scores:
            file.write(f"{score}\n")

def assess():
    assessment = {
        "revision_id": 1, 
        "reference_id": 1, 
        "type": "semantic-similarity"
    }

    if isinstance(assessment, dict):
        assessment = Assessment(**assessment)
        
    # Paths to text files on the system
    revision_file_path = "./data/aak-aak-small.txt" 
    reference_file_path = "./data/aai-aai-small.txt"
    
    revision_text = get_text(revision_file_path)
    reference_text = get_text(reference_file_path)

    # Create DataFrames from text files
    revision_lines = revision_text.split('\n')
    reference_lines = reference_text.split('\n')

    revision_df = pd.DataFrame(revision_lines, columns=["revision"])
    reference_df = pd.DataFrame(reference_lines, columns=["reference"])

    # Merge the DataFrames (assuming you want to concatenate them along the columns)
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    semsim_model, semsim_tokenizer = get_labse_model()
    semsim_model.to(device)
    sim_scores = []
    
    for i, (rev_batch, ref_batch) in enumerate(tqdm(zip(rev_sents_batched, ref_sents_batched), total=len(rev_sents_batched))):
        # Debugging: Ensure no empty strings are in the batch
        rev_batch = [sent if sent else " " for sent in rev_batch]
        ref_batch = [sent if sent else " " for sent in ref_batch]
        
        try:
            sim_scores.extend(get_sim_scores(rev_batch, ref_batch, semsim_model, semsim_tokenizer, device=device))
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

    print(results[:20])

    return {"results": results}

def main():
    assess()

if __name__ == "__main__":
    main()