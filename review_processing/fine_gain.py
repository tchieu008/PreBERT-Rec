import numpy as np
import torch
import os
from tqdm import tqdm
from nltk.corpus import sentiwordnet as swn
from sklearn.cluster import KMeans, Birch, DBSCAN, MeanShift, BisectingKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer

from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from nltk.corpus import wordnet as wn
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from helper.general_functions import preprocessed, word_segment


class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,  # Cắt bớt các chuỗi dài hơn max_len
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def collate_fn(tokenizer):
    def collate_batch(batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        labels = [item['labels'] for item in batch]
        
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        
        labels = torch.tensor(labels, dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    return collate_batch

def fine_tune_bert(texts, labels, num_labels, epochs=50, batch_size=8, max_len=512, learning_rate=2e-5, save_dir='./chkpt'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = CustomDataset(texts, labels, tokenizer, max_len)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn(tokenizer))
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    model = model.to(device) 

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    start_epoch = 0
    checkpoint_path = os.path.join(save_dir, "bert_last_checkpoint.pt")

    if os.path.exists(checkpoint_path):
        print(f"Checkpoint found at {checkpoint_path}. Loading checkpoint.")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming training from epoch {start_epoch + 1}")
        return model, tokenizer

    # Training loop
    for epoch in range(start_epoch, epochs):  # Bắt đầu từ epoch đã lưu
        model.train()
        total_loss = 0
        progress_bar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix({"Loss": total_loss / len(progress_bar)})

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss / len(loader)
        }, checkpoint_path)
        
        print(f"Epoch {epoch + 1} complete. Model saved to {checkpoint_path}.")

    print(f"Training complete. Final loss: {total_loss / len(loader):.4f}")

    return model, tokenizer

def get_bert_embeddings(texts, tokenizer, model, device):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model.bert(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

def get_tbert_model(data_df, split_data, num_topics, num_words, cluster_method='Kmeans'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cleaned_data = data_df.dropna(subset=['filteredReviewText', 'overall_new'])
    cleaned_data['overall_new'] = cleaned_data['overall_new'].apply(lambda x: x - 1)

    texts = cleaned_data['filteredReviewText'].tolist()
    labels = cleaned_data['overall_new'].tolist()

    model, tokenizer = fine_tune_bert(texts, labels, num_labels=5, epochs=10)
    model = model.to(device)
    
    embeddings = []
    for text in split_data:
        embedding = get_bert_embeddings([' '.join(text)], tokenizer, model, device)
        embeddings.append(embedding)

    embeddings = torch.vstack(embeddings).to(device)

    # Clustering
    if cluster_method == 'Kmeans':
        print("Kmeans")
        clustering = KMeans(n_clusters=num_topics, random_state=42).fit(embeddings.cpu().numpy())
    elif cluster_method == 'Birch':
        print("Birch")
        clustering = Birch(n_clusters=num_topics).fit(embeddings.cpu().numpy())
    elif cluster_method == 'DBSCAN':
        print("DBSCAN")
        clustering = DBSCAN(eps=3, min_samples=num_topics).fit(embeddings.cpu().numpy())
    elif cluster_method == 'MeanShift':
        print("MeanShift")
        clustering = MeanShift(bandwidth=num_topics).fit(embeddings.cpu().numpy())
    elif cluster_method == 'BisectingKMeans':
        print("BisectingKMeans")
        clustering = BisectingKMeans(n_clusters=num_topics, random_state=42).fit(embeddings.cpu().numpy())

    labels = clustering.labels_

    topic_to_words = []
    for i in range(num_topics):
        cluster_indices = [j for j, label in enumerate(labels) if label == i]
        cluster_texts = [' '.join(split_data[j]) for j in cluster_indices]
        cluster_texts = [text for text in cluster_texts if text.strip()]

        if not cluster_texts:
            print(f"No valid texts in cluster {i}, skipping.")
            topic_to_words.append([])
            continue
        
        vectorizer = TfidfVectorizer(max_features=num_words, stop_words='english')
        
        try:
            tfidf_matrix = vectorizer.fit_transform(cluster_texts)
            if tfidf_matrix.shape[1] == 0:  # No valid words were found after stop words removal
                print(f"Cluster {i} contains only stop words or empty texts.")
                topic_to_words.append([])
                continue
        except ValueError as e:
            print(f"Error processing cluster {i}: {e}")
            topic_to_words.append([])
            continue

        indices = np.argsort(tfidf_matrix.sum(axis=0)).flatten()[::-1]
        feature_names = vectorizer.get_feature_names_out()
        top_words = [feature_names[ind] for ind in indices[:num_words]]
        topic_to_words.append(top_words)
    # print("topic_to_words: ", topic_to_words)
    
    return model, tokenizer, topic_to_words


analyzer = SentimentIntensityAnalyzer()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Sử dụng GPU nếu có

def get_word_sentiment_score_by_vader(word):
    sentiment_dict = analyzer.polarity_scores(word)
    return sentiment_dict['compound']

def get_top_synonyms(word, top_n=4):
    noun_synsets = wn.synsets(word, pos=wn.NOUN)
    adj_synsets = wn.synsets(word, pos=wn.ADJ)
    
    all_synsets = noun_synsets + adj_synsets
    synonym_scores = []
    for synset in all_synsets:
        for lemma in synset.lemma_names():
            if lemma.lower() != word.lower() and lemma not in [syn[0] for syn in synonym_scores]:
                synonym_scores.append((lemma))
    
    return synonym_scores[:top_n]

def get_word_sentiment_score(word):
    m = list(swn.senti_synsets(word))
    s = 0
    if not m:
        return s  # Trả về 0 nếu không tìm thấy synset nào cho từ này
    for synset in m:
        s += get_word_sentiment_score_by_vader(synset.synset.name().split('.')[0])
    return s

def get_synonyms_sentiment_scores(word, top_n=4):
    synonyms = get_top_synonyms(word, top_n=top_n)
    scores = 0
    
    for synonym in synonyms:
        sentiment_score = get_word_sentiment_score(synonym)
        scores += sentiment_score

    scores = scores / top_n
    return scores

def get_topic_sentiment_matrix_tbert(text, topic_word_matrix, dependency_parser, topic_nums=50):
    topic_sentiment_m = torch.zeros(topic_nums, device=device)
    try:
        sentences = preprocessed(text)
        dep_parser_result_p = []
        
        for i in sentences:
            dep_parser_result = dependency_parser.raw_parse(i)
            for j in dep_parser_result:
                dep_parser_result_p.append([j[0][0], j[2][0]])
                
        for topic_id, cur_topic_words in enumerate(topic_word_matrix):
            cur_topic_senti_word = []
            for word in word_segment(text):
                if any(word in sublist for sublist in cur_topic_words):
                    cur_topic_senti_word.append(word)
                    for p in dep_parser_result_p:
                        if p[0] == word:
                            cur_topic_senti_word.append(p[1])
                        if p[1] == word:
                            cur_topic_senti_word.append(p[0])

            if cur_topic_senti_word: 
                cur_topic_sentiment = sum(get_synonyms_sentiment_scores(senti_word) for senti_word in cur_topic_senti_word)
                topic_sentiment_m[topic_id] = torch.tensor(np.clip(cur_topic_sentiment, -5, 5), device=device)
            else:
                topic_sentiment_m[topic_id] = torch.tensor(0, device=device) 
                
        return topic_sentiment_m
    except Exception as e:
        print("get_topic_sentiment_matrix_tbert's error: ", e, " text: ", text)
        return topic_sentiment_m



