import os, re
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from gensim import corpora, models
import pyLDAvis.gensim
import pyLDAvis.gensim_models
import pyLDAvis
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
from nrclex import NRCLex
import spacy
import numpy as np
import seaborn as sns
from sentence_transformers import SentenceTransformer

nltk.download('stopwords')
nltk.download('punkt')

# Step 1: Read all text files
data_path = 'data'
authors = {}
for file in os.listdir(data_path):
    if file.endswith('.txt'):
        author_name = file.replace('.txt', '').capitalize()
        with open(os.path.join(data_path, file), 'r', encoding='utf-8') as f:
            poems = f.read().split('###')
            authors[author_name] = [p.strip() for p in poems if len(p.strip()) > 0]

# Step 2: Basic Cleaning
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = [w.lower() for w in text.split() if w.lower() not in stop_words]
    return ' '.join(words)

for author in authors:
    authors[author] = [clean_text(p) for p in authors[author]]

# Step 3: Sentiment Analysis
sentiment_scores = {}
for author, poems in authors.items():
    scores = [TextBlob(p).sentiment.polarity for p in poems]
    sentiment_scores[author] = scores

# Convert to DataFrame
sentiment_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in sentiment_scores.items()]))

# Plot Sentiment Comparison
sentiment_df.boxplot()
plt.title("Sentiment Comparison Across Authors")
plt.ylabel("Polarity (-1 Negative → +1 Positive)")
plt.show()

# Step 4: WordCloud for Each Author
for author, poems in authors.items():
    text = ' '.join(poems)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"WordCloud of {author}")
    plt.show()

# Step 5: TF-IDF Thematic Similarity Analysis


# Combine all poems for each author into one document
author_docs = {author: ' '.join(poems) for author, poems in authors.items()}

# Create TF-IDF matrix
vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = vectorizer.fit_transform(author_docs.values())

# Compute cosine similarity
similarity_matrix = cosine_similarity(tfidf_matrix)

# Convert to DataFrame
authors_list = list(author_docs.keys())
similarity_df = pd.DataFrame(similarity_matrix, index=authors_list, columns=authors_list)

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(similarity_df, annot=True, cmap="YlGnBu")
plt.title("TF-IDF Based Thematic Similarity Between Authors")
plt.show()

# Step 6: Prepare Data for Topic Modeling
texts = []
author_labels = []

for author, poems in authors.items():
    tokenized_poems = [p.split() for p in poems]
    texts.extend(tokenized_poems)
    author_labels.extend([author] * len(poems))

# Create Dictionary and Corpus
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# Step 7: Train LDA Model
num_topics = 5  # You can adjust between 3–6 depending on variety
lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15, random_state=42)

topic_dist = {}

for author, poems in authors.items():
    # Convert each poem to bag-of-words
    bow_docs = [dictionary.doc2bow(p.split()) for p in poems]
    
    # Get topic distribution for each poem and average them
    author_topic_matrix = []

    for bow in bow_docs:
        topic_probs = lda_model.get_document_topics(bow, minimum_probability=0)
        topic_probs = [prob for _, prob in topic_probs]
        author_topic_matrix.append(topic_probs)

    # Average over all poems of the author
    author_topic_avg = np.mean(author_topic_matrix, axis=0)
    topic_dist[author] = author_topic_avg

# Convert to DataFrame for heatmap
lda_topic_df = pd.DataFrame(topic_dist)
lda_topic_df.index = [f"Topic {i+1}" for i in range(len(lda_topic_df))]

# --- Plot Heatmap ---
plt.figure(figsize=(8, 6))
sns.heatmap(lda_topic_df, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("LDA Topic Distribution × Authors")
plt.xlabel("Authors")
plt.ylabel("Topics")
plt.tight_layout()
plt.show()

# Display Topics
print("\n--- LDA Topics Discovered ---\n")
for idx, topic in lda_model.print_topics(num_topics=num_topics, num_words=8):
    print(f"Topic {idx+1}: {topic}")

# Step 8: Visualize Topics
#lda_vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
#pyLDAvis.save_html(lda_vis, "lda_topics.html")

print("\n✅ LDA visualization saved as 'lda_topics.html' in your project folder.")
print("👉 You can open it in your browser to explore topics interactively.")

# Step 8: Deep Semantic Analysis with Sentence-BERT


# Load the pre-trained Sentence-BERT model
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')  # lighter, faster


# Prepare data for embeddings
poem_texts = []
poem_authors = []

for author, poems in authors.items():
    for poem in poems:
        poem_texts.append(poem)
        poem_authors.append(author)

# Generate embeddings for all poems
embeddings = model.encode(poem_texts)

# Step: Generate Sentence Embeddings (for each poem)
model = SentenceTransformer('all-MiniLM-L6-v2')

all_poems = []
author_labels = []

for author, poems in authors.items():
    for p in poems:
        all_poems.append(p)
        author_labels.append(author)

sentence_embeddings = model.encode(all_poems, convert_to_tensor=False)

# Reduce dimensions for visualization using PCA
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# Create a DataFrame for visualization
semantic_df = pd.DataFrame({
    'Author': poem_authors,
    'PC1': reduced_embeddings[:, 0],
    'PC2': reduced_embeddings[:, 1]
})

# Plot
plt.figure(figsize=(10, 7))
for author in semantic_df['Author'].unique():
    subset = semantic_df[semantic_df['Author'] == author]
    plt.scatter(subset['PC1'], subset['PC2'], label=author, s=80)
plt.title("Semantic Similarity Map (Sentence-BERT PCA)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()

# Step 6: Compute Author-to-Author Similarity
author_embeddings = {}
for author in authors.keys():
    combined_text = ' '.join(authors[author])
    author_embeddings[author] = model.encode(combined_text)

# Compute cosine similarity between authors
sim_matrix = pd.DataFrame(cosine_similarity(list(author_embeddings.values())),
                          index=author_embeddings.keys(),
                          columns=author_embeddings.keys())

print("\n--- Author-to-Author Semantic Similarity ---")
print(sim_matrix)

# Sentence embeddings already computed: sentence_embeddings
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
tsne_results = tsne.fit_transform(sentence_embeddings)

plt.figure(figsize=(10, 6))
for author, color in zip(['Dickinson', 'Shakespeare', 'Tagore'], ['blue', 'orange', 'green']):
    idx = [i for i, a in enumerate(author_labels) if a == author]
    plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1], c=color, label=author, alpha=0.6)

plt.title("t-SNE Semantic Map of Authors")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend()
plt.show()

Z = linkage(similarity_df, method='ward')

plt.figure(figsize=(8, 5))
dendrogram(Z, labels=similarity_df.index)
plt.title("Author Hierarchical Clustering Dendrogram")
plt.xlabel("Authors")
plt.ylabel("Distance")
plt.show()

emotion_scores = {'Dickinson': [], 'Shakespeare': [], 'Tagore': []}

for author, poems in authors.items():
    combined = ' '.join(poems)
    emo = NRCLex(combined)
    emotion_scores[author] = emo.raw_emotion_scores

emotion_df = pd.DataFrame(emotion_scores).fillna(0)

plt.figure(figsize=(8, 6))
sns.heatmap(emotion_df, annot=True, cmap='coolwarm')
plt.title("Emotion Heatmap Across Authors")
plt.show()

nlp = spacy.load("en_core_web_sm")

pos_counts = {'Dickinson': {}, 'Shakespeare': {}, 'Tagore': {}}

for author, poems in authors.items():
    text = ' '.join(poems)
    doc = nlp(text)
    
    for token in doc:
        if token.pos_ not in pos_counts[author]:
            pos_counts[author][token.pos_] = 0
        pos_counts[author][token.pos_] += 1

pos_df = pd.DataFrame(pos_counts).fillna(0)

pos_df.plot(kind='bar', figsize=(12, 6))
plt.title("Part-of-Speech (POS) Distribution Across Authors")
plt.xlabel("POS Type")
plt.ylabel("Frequency")
plt.show()