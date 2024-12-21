import os
import tqdm
import torch
import pandas as pd
from helper.general_functions import create_and_write_csv, load_data_from_csv, split_text
from init import dep_parser
from review_processing.coarse_gain import get_coarse_sentiment_score
from review_processing.fine_gain import get_tbert_model, get_topic_sentiment_matrix_tbert


def merge_fine_coarse_features(data_df, num_factors, groupBy="reviewerID"):
    feature_dict = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for id, df in data_df.groupby(groupBy):
        feature = torch.zeros(num_factors, device=device)
        list_finefeature = df['fine_feature']
        list_coarse_feature = df['coarse_feature']
        
        for fine, coarse in zip(list_finefeature, list_coarse_feature):
            try:
                fine_feature = torch.tensor([float(x) for x in fine.strip('[]').split()], device=device)
                coarse_feature = torch.tensor(float(coarse), device=device)  
                feature += fine_feature * coarse_feature
            except Exception as e:
                print("Error: ", e)
                continue
        feature_dict[id] = feature.cpu().numpy() 
        
    return feature_dict

# Extract fine-grained and coarse-grained features
def extract_review_feature(data_df, model, dep_parser, tokenizer, topic_word_matrix, num_topics):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = model.to(device)  

    row_list = []
    print("data_train_size: ", data_df.shape[0])
    for asin, df in tqdm.tqdm(data_df.groupby("asin")):
        review_text = df["filteredReviewText"].tolist()
        overall = df["overall_new"].tolist()
        reviewerID = df["reviewerID"].tolist()

        for i, text in enumerate(review_text):
            try:
                # Convert text về chuỗi rỗng nếu nó là None
                if text is None:
                    text = ""
                fine_feature = torch.zeros(num_topics, device=device)  # Giữ tensor trên GPU
                coarse_feature = 0

                text_chunks = split_text(text) if text else [""]
                count_null = 0
                for chunk in text_chunks:
                    if chunk and chunk.strip():
                        try:
                            # Giữ tensor trên GPU trong quá trình tính toán
                            fine_feature_chunk = get_topic_sentiment_matrix_tbert(chunk, topic_word_matrix, dep_parser, topic_nums=num_topics)
                            coarse_feature_chunk = get_coarse_sentiment_score(model, tokenizer, chunk)
                        except KeyError as e:
                            print(f"Skipping chunk due to missing key in vocabulary: {e}")
                            continue
                    else:
                        count_null += 1
                        continue

                    fine_feature += fine_feature_chunk
                    coarse_feature += coarse_feature_chunk

                coarse_feature /= max(1, len(text_chunks) - count_null)
                fine_feature = torch.clamp(fine_feature, min=-5, max=5)

                new_row = {
                    'reviewerID': reviewerID[i], 
                    'itemID': asin, 
                    'overall': overall[i],
                    'fine_feature': fine_feature.cpu().numpy(),
                    'coarse_feature': coarse_feature
                }
                row_list.append(new_row)
            except Exception as e:
                print(f"Error: {e}, Text: {text}, fine_feature: {fine_feature}")
                continue

    return pd.DataFrame(row_list, columns=['reviewerID', 'itemID', 'overall', 'fine_feature', 'coarse_feature'])


# Global variables to store features
reviewer_feature_dict = {}
item_feature_dict = {}
allFeatureReview = pd.DataFrame(columns=['reviewerID', 'itemID', 'overall', 'unixReviewTime', 'fine_feature', 'coarse_feature'])

def initialize_features(filename, num_factors):
    # print("Initialize features")
    global reviewer_feature_dict, item_feature_dict
    allreviews_path = "feature/allFeatureReview_"
    reviewer_path = "/feature/reviewer_feature_"
    item_path = "feature/item_feature_"
    
    # Initialize or load reviewer features
    if os.path.exists(reviewer_path + filename +".csv"):
        reviewer_feature_dict = load_data_from_csv(reviewer_path + filename +".csv")
    else:
        allFeatureReview = pd.read_csv(allreviews_path + filename +".csv")
        reviewer_feature_dict = merge_fine_coarse_features(allFeatureReview, num_factors, groupBy="reviewerID")
        create_and_write_csv("reviewer_feature_" + filename, reviewer_feature_dict)
        
    # Initialize or load item features
    if os.path.exists(item_path+ filename +".csv"):
        item_feature_dict = load_data_from_csv(item_path+ filename +".csv")
    else:
        allFeatureReview = pd.read_csv(allreviews_path+ filename +".csv")
        item_feature_dict = merge_fine_coarse_features(allFeatureReview, num_factors, groupBy="itemID")
        create_and_write_csv("item_feature_" + filename, item_feature_dict)
    return reviewer_feature_dict, item_feature_dict
        
def extract_features(data_df, split_data, num_topics, num_words, filename):
    allreviews_path = "feature/allFeatureReview_"
    if os.path.exists(allreviews_path + filename +".csv"):
        allFeatureReview = pd.read_csv(allreviews_path + filename +".csv")
    else:
        model, tokenizer, topic_word_matrix = get_tbert_model(data_df, split_data, num_topics, num_words, cluster_method="Birch")
        allFeatureReview = extract_review_feature(data_df, model, dep_parser, tokenizer, topic_word_matrix, num_topics)
        allFeatureReview.to_csv(allreviews_path + filename +".csv", index=False)
    return allFeatureReview
