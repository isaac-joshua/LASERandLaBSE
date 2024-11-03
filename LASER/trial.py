import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import torch
from laserembeddings import Laser
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LinearRegression
from collections import Counter
from transformers import pipeline
from scipy.stats import ttest_ind
import nltk
import threading
from tqdm import tqdm
import sys

# Download NLTK data files (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')

class SimilarityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Semantic Similarity Analysis")

        # File selection buttons
        self.select_revision_button = tk.Button(root, text="Select Revision File", command=self.select_revision_file)
        self.select_revision_button.pack()

        self.select_reference_button = tk.Button(root, text="Select Reference File", command=self.select_reference_file)
        self.select_reference_button.pack()

        # Text output display
        self.text_output = tk.Text(root, height=15, width=100)
        self.text_output.pack()

        # Figure for plotting with multiple subplots
        self.figure, self.axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack()

        # Initialize file paths
        self.revision_file_path = None
        self.reference_file_path = None

        # Initialize NER pipeline
        # self.ner_pipeline = pipeline("ner", grouped_entities=True, device=0 if torch.cuda.is_available() else -1)
        self.ner_pipeline = pipeline("ner", model="xlm-roberta-large-finetuned-conll03-english", grouped_entities=True, device=0 if torch.cuda.is_available() else -1)

    def select_revision_file(self):
        self.revision_file_path = filedialog.askopenfilename(title="Select the Revision File",
                                                             filetypes=[("Text Files", "*.txt")])
        self.text_output.insert(tk.END, f"Selected Revision File: {self.revision_file_path}\n")

    def select_reference_file(self):
        self.reference_file_path = filedialog.askopenfilename(title="Select the Reference File",
                                                              filetypes=[("Text Files", "*.txt")])
        self.text_output.insert(tk.END, f"Selected Reference File: {self.reference_file_path}\n")

        if self.revision_file_path and self.reference_file_path:
            # Run analysis in a separate thread
            analysis_thread = threading.Thread(target=self.run_analysis)
            analysis_thread.start()

    def run_analysis(self):
        try:
            revision_text = self.get_text(self.revision_file_path)
            reference_text = self.get_text(self.reference_file_path)

            revision_sentences = revision_text.split('\n')
            reference_sentences = reference_text.split('\n')

            # Ensure both lists have the same length
            min_length = min(len(revision_sentences), len(reference_sentences))
            revision_sentences = revision_sentences[:min_length]
            reference_sentences = reference_sentences[:min_length]

            # Limit the number of sentences for testing
            # max_sentences = 100  # Uncomment and adjust for testing
            # revision_sentences = revision_sentences[:max_sentences]
            # reference_sentences = reference_sentences[:max_sentences]

            sim_scores, embeddings = self.get_sim_scores_and_embeddings(revision_sentences, reference_sentences)

            # Detect names in revision sentences
            names_presence = self.detect_names_in_sentences(revision_sentences)

            # Display descriptive statistics in the UI
            self.display_descriptive_statistics(sim_scores)

            # Plot similarity score distribution
            self.plot_similarity_distribution(sim_scores)

            # Perform clustering and plot clusters
            labels = self.cluster_verses_embeddings(embeddings)
            self.characterize_clusters(revision_sentences, labels)

            # Plot similarity scores with names highlighted
            self.plot_similarity_with_names(sim_scores, names_presence)

            # Analyze extreme cases in UI
            self.analyze_extreme_cases(revision_sentences, reference_sentences, sim_scores)

            # Perform regression analysis and plot
            self.regression_analysis(embeddings, sim_scores)

            # Statistical analysis
            self.compare_similarity_with_names(sim_scores, names_presence)

        except Exception as e:
            print(f"An error occurred: {e}", file=sys.stderr)
            self.text_output.insert(tk.END, f"An error occurred: {e}\n")

    def get_text(self, file_path):
        encodings = ['utf-8', 'latin-1', 'ascii', 'utf-16']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    content = file.read()
                return content
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Unable to read the file with any of the encodings: {encodings}")

    def get_sim_scores_and_embeddings(self, rev_sents_output, ref_sents_output):
        laser = Laser()
        batch_size = 32  # Adjust as needed
        rev_embeddings = []
        ref_embeddings = []
        for i in tqdm(range(0, len(rev_sents_output), batch_size), desc="Embedding Sentences"):
            rev_batch = rev_sents_output[i:i+batch_size]
            ref_batch = ref_sents_output[i:i+batch_size]
            rev_embeddings_batch = laser.embed_sentences(rev_batch, lang='en')
            ref_embeddings_batch = laser.embed_sentences(ref_batch, lang='en')
            rev_embeddings.append(rev_embeddings_batch)
            ref_embeddings.append(ref_embeddings_batch)
        rev_sents_embedding = np.vstack(rev_embeddings)
        ref_sents_embedding = np.vstack(ref_embeddings)
        sim_scores = torch.nn.functional.cosine_similarity(
            torch.tensor(rev_sents_embedding),
            torch.tensor(ref_sents_embedding),
            dim=1
        ).tolist()
        return sim_scores, rev_sents_embedding

    def detect_names_in_sentences(self, sentences):
        names_presence = []
        batch_size = 16  # Adjust based on your system's capabilities
        for i in tqdm(range(0, len(sentences), batch_size), desc="Processing NER in Batches"):
            batch_sentences = sentences[i:i+batch_size]
            ner_results_batch = self.ner_pipeline(batch_sentences)
            for ner_results in ner_results_batch:
                has_name = any(ent['entity_group'] == 'PER' for ent in ner_results)
                names_presence.append(has_name)
        return names_presence

    def display_descriptive_statistics(self, sim_scores):
        mean_score = np.mean(sim_scores)
        median_score = np.median(sim_scores)
        std_dev_score = np.std(sim_scores)
        self.text_output.insert(tk.END, f"\nDescriptive Statistics:\n")
        self.text_output.insert(tk.END, f"Mean similarity: {mean_score:.4f}\n")
        self.text_output.insert(tk.END, f"Median similarity: {median_score:.4f}\n")
        self.text_output.insert(tk.END, f"Standard Deviation: {std_dev_score:.4f}\n\n")

    def plot_similarity_distribution(self, sim_scores):
        ax = self.axes[0]
        ax.clear()
        ax.hist(sim_scores, bins=20, alpha=0.75, color='skyblue', edgecolor='black')
        ax.set_title('Distribution of Similarity Scores')
        ax.set_xlabel('Similarity Score')
        ax.set_ylabel('Frequency')
        self.canvas.draw()

    def cluster_verses_embeddings(self, embeddings):
        optimal_k = 2
        optimal_silhouette = -1
        for k in range(2, 10):
            kmeans = KMeans(n_clusters=k, random_state=0).fit(embeddings)
            silhouette_avg = silhouette_score(embeddings, kmeans.labels_)
            # Print silhouette scores for different k values
            print(f"Silhouette Score for k={k}: {silhouette_avg}")
            if silhouette_avg > optimal_silhouette:
                optimal_k = k
                optimal_silhouette = silhouette_avg
        kmeans = KMeans(n_clusters=optimal_k, random_state=0).fit(embeddings)
        self.text_output.insert(tk.END, f"Optimal number of clusters: {optimal_k}\n\n")
        return kmeans.labels_

    def characterize_clusters(self, sentences, labels):
        stop_words = set(stopwords.words('english'))
        cluster_contents = {i: [] for i in set(labels)}
        for sentence, label in zip(sentences, labels):
            words = word_tokenize(sentence)
            filtered_words = [word.lower() for word in words if word.lower() not in stop_words and word.isalnum()]
            cluster_contents[label].extend(filtered_words)

        for label, words in cluster_contents.items():
            word_freq = Counter(words)
            self.text_output.insert(tk.END, f"Most common words in cluster {label}:\n")
            common_words = ', '.join([f"{word} ({count})" for word, count in word_freq.most_common(10)])
            self.text_output.insert(tk.END, f"{common_words}\n\n")

    def plot_similarity_with_names(self, sim_scores, names_presence):
        ax = self.axes[1]
        ax.clear()
        x_values = range(len(sim_scores))
        ax.plot(x_values, sim_scores, label='Similarity Score', color='blue')

        # Highlight verses with names
        names_indices = [i for i, has_name in enumerate(names_presence) if has_name]
        names_scores = [sim_scores[i] for i in names_indices]
        ax.scatter(names_indices, names_scores, color='red', label='Verses with Names', marker='o')

        ax.set_title('Similarity Scores with Names Highlighted')
        ax.set_xlabel('Verse Index')
        ax.set_ylabel('Similarity Score')
        ax.legend()
        self.canvas.draw()

    def analyze_extreme_cases(self, revision_sentences, reference_sentences, sim_scores, num_cases=5):
        sorted_indices = np.argsort(sim_scores)
        self.text_output.insert(tk.END, f"\nTop {num_cases} Lowest Similarity Verses:\n")
        for i in sorted_indices[:num_cases]:
            self.text_output.insert(tk.END, f"Index {i}:\n")
            self.text_output.insert(tk.END, f"Revision: {revision_sentences[i]}\n")
            self.text_output.insert(tk.END, f"Reference: {reference_sentences[i]}\n")
            self.text_output.insert(tk.END, f"Score: {sim_scores[i]:.4f}\n\n")

        self.text_output.insert(tk.END, f"\nTop {num_cases} Highest Similarity Verses:\n")
        for i in sorted_indices[-num_cases:][::-1]:
            self.text_output.insert(tk.END, f"Index {i}:\n")
            self.text_output.insert(tk.END, f"Revision: {revision_sentences[i]}\n")
            self.text_output.insert(tk.END, f"Reference: {reference_sentences[i]}\n")
            self.text_output.insert(tk.END, f"Score: {sim_scores[i]:.4f}\n\n")

    def regression_analysis(self, embeddings, sim_scores):
        ax = self.axes[2]
        model = LinearRegression()
        model.fit(embeddings, sim_scores)
        predictions = model.predict(embeddings)
        ax.clear()
        ax.scatter(sim_scores, predictions, alpha=0.5, edgecolors='k')
        ax.set_xlabel('Actual Scores')
        ax.set_ylabel('Predicted Scores')
        ax.set_title('Regression Analysis Results')
        # Plotting the line y=x for reference
        min_score = min(sim_scores + predictions.tolist())
        max_score = max(sim_scores + predictions.tolist())
        ax.plot([min_score, max_score], [min_score, max_score], color='red', linestyle='--', label='Ideal Fit')
        ax.legend()
        self.canvas.draw()

    def compare_similarity_with_names(self, sim_scores, names_presence):
        scores_with_names = [score for score, has_name in zip(sim_scores, names_presence) if has_name]
        scores_without_names = [score for score, has_name in zip(sim_scores, names_presence) if not has_name]
        t_stat, p_value = ttest_ind(scores_with_names, scores_without_names, equal_var=False)
        self.text_output.insert(tk.END, f"\nSimilarity Scores Analysis Based on Presence of Names:\n")
        self.text_output.insert(tk.END, f"Average similarity (with names): {np.mean(scores_with_names):.4f}\n")
        self.text_output.insert(tk.END, f"Average similarity (without names): {np.mean(scores_without_names):.4f}\n")
        self.text_output.insert(tk.END, f"T-test p-value: {p_value:.4f}\n\n")

    def plot_time_series(self, sim_scores):
        # This function is no longer used since we modified the plot in plot_similarity_with_names
        pass

if __name__ == "__main__":
    root = tk.Tk()
    app = SimilarityApp(root)
    root.mainloop()
