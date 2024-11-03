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
import gc
import torch
gc.collect()

torch.cuda.empty_cache()