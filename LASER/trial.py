import os
import tkinter as tk
from tkinter import filedialog, Text
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import torch
import numpy as np
from laserembeddings import Laser
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.linear_model import LinearRegression

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
        self.text_output = Text(root, height=20, width=80)
        self.text_output.pack()

        # Figure for plotting
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack()

        # Initialize file paths
        self.revision_file_path = None
        self.reference_file_path = None

    def select_revision_file(self):
        self.revision_file_path = filedialog.askopenfilename(title="Select the Revision File",
                                                             filetypes=[("Text Files", "*.txt")])
        self.text_output.insert(tk.END, f"Selected Revision File: {self.revision_file_path}\n")

    def select_reference_file(self):
        self.reference_file_path = filedialog.askopenfilename(title="Select the Reference File",
                                                              filetypes=[("Text Files", "*.txt")])
        self.text_output.insert(tk.END, f"Selected Reference File: {self.reference_file_path}\n")

        if self.revision_file_path and self.reference_file_path:
            self.run_analysis()

    def run_analysis(self):
        # Generate filenames for saving results
        revision_filename = os.path.basename(self.revision_file_path).split('-')[0]
        reference_filename = os.path.basename(self.reference_file_path).split('-')[0]

        # Load and process files
        revision_text = self.get_text(self.revision_file_path)
        reference_text = self.get_text(self.reference_file_path)

        revision_sentences = revision_text.split('\n')
        reference_sentences = reference_text.split('\n')

        sim_scores, embeddings = self.get_sim_scores_and_embeddings(revision_sentences, reference_sentences)

        # Display descriptive statistics
        self.display_descriptive_statistics(sim_scores)

        # Plot similarity score distribution
        self.plot_similarity_distribution(sim_scores)

        # Perform clustering
        labels = self.cluster_verses_embeddings(embeddings)

        # Display clustering results
        self.characterize_clusters(revision_sentences, labels)

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
        rev_sents_embedding = laser.embed_sentences(rev_sents_output, lang='en')
        ref_sents_embedding = laser.embed_sentences(ref_sents_output, lang='en')
        sim_scores = torch.nn.functional.cosine_similarity(
            torch.tensor(rev_sents_embedding),
            torch.tensor(ref_sents_embedding),
            dim=1
        ).tolist()
        return sim_scores, rev_sents_embedding

    def display_descriptive_statistics(self, sim_scores):
        mean_score = np.mean(sim_scores)
        median_score = np.median(sim_scores)
        std_dev_score = np.std(sim_scores)
        self.text_output.insert(tk.END, f"Mean similarity: {mean_score}\n")
        self.text_output.insert(tk.END, f"Median similarity: {median_score}\n")
        self.text_output.insert(tk.END, f"Standard Deviation of similarity scores: {std_dev_score}\n")

    def plot_similarity_distribution(self, sim_scores):
        self.ax.clear()
        self.ax.hist(sim_scores, bins=20, alpha=0.75)
        self.ax.set_title('Distribution of Similarity Scores')
        self.ax.set_xlabel('Similarity Score')
        self.ax.set_ylabel('Frequency')
        self.canvas.draw()

    def cluster_verses_embeddings(self, embeddings):
        optimal_k = 2
        optimal_silhouette = -1
        for k in range(2, 10):
            kmeans = KMeans(n_clusters=k, random_state=0).fit(embeddings)
            silhouette_avg = silhouette_score(embeddings, kmeans.labels_)
            self.text_output.insert(tk.END, f"Silhouette Score for k={k}: {silhouette_avg}\n")
            if silhouette_avg > optimal_silhouette:
                optimal_k = k
                optimal_silhouette = silhouette_avg
        kmeans = KMeans(n_clusters=optimal_k, random_state=0).fit(embeddings)
        self.text_output.insert(tk.END, f"Optimal number of clusters: {optimal_k}\n")
        return kmeans.labels_

    def characterize_clusters(self, sentences, labels):
        stop_words = set(stopwords.words('english'))
        cluster_contents = {i: [] for i in set(labels)}
        for sentence, label in zip(sentences, labels):
            words = word_tokenize(sentence)
            filtered_words = [word for word in words if word not in stop_words and word.isalnum()]
            cluster_contents[label].extend(filtered_words)
        
        for label, words in cluster_contents.items():
            word_freq = Counter(words)
            self.text_output.insert(tk.END, f"Most common words in cluster {label}: {word_freq.most_common(10)}\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = SimilarityApp(root)
    root.mainloop()
