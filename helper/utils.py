import json
import numpy as np
import pandas as pd
import nltk
from sklearn.calibration import LabelEncoder
import torch
import json
from torch.utils.data import DataLoader, TensorDataset

from helper.general_functions import format_array, parse_array_from_string
nltk.download('punkt')
nltk.download('stopwords')


def read_data(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            try:
                raw_sample = json.loads(line)
                if 'reviewText' not in raw_sample:
                    raw_sample['reviewText'] = ''
                data.append([raw_sample['reviewerID'],
                             raw_sample['asin'],
                             raw_sample['overall'],
                             raw_sample['overall_new'],
                             raw_sample['reviewText'],
                             raw_sample['filteredReviewText']])
            except json.JSONDecodeError:
                pass
    return data

def csv_to_dataloader(csv_link, batch_size, shuffle=True):
    df = pd.read_csv(csv_link)

    df['Udeep'] = df['Udeep'].apply(parse_array_from_string)
    df['Ideep'] = df['Ideep'].apply(parse_array_from_string)
    
    df['reviewerID'] = df['reviewerID'].astype(int)
    df['itemID'] = df['itemID'].astype(int)
    
    reviewerID_tensor = torch.tensor(df['reviewerID'].values, dtype=torch.long)
    itemID_tensor = torch.tensor(df['itemID'].values, dtype=torch.long)
    overall_tensor = torch.tensor(df['overall'].values, dtype=torch.float32)
    Udeep_tensor = torch.tensor(df['Udeep'].tolist(), dtype=torch.float32)
    Ideep_tensor = torch.tensor(df['Ideep'].tolist(), dtype=torch.float32)
    itembias_tensor = torch.tensor(df['item_bias'].values, dtype=torch.float32)
    userbias_tensor = torch.tensor(df['user_bias'].values, dtype=torch.float32)
    
    dataset = TensorDataset(reviewerID_tensor, itemID_tensor, overall_tensor, Udeep_tensor, Ideep_tensor, itembias_tensor, userbias_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=shuffle)
    
    return dataloader


def map_and_add_column(df1, df2, column_df1, column_df2=None, column_to_map=None, new_column_name='new_column'):
    if isinstance(df2, pd.DataFrame):
        if column_df2 is None or column_to_map is None:
            raise ValueError("Cần chỉ định column_df2 và column_to_map khi df2 là DataFrame")
        map_dict = df2.set_index(column_df2)[column_to_map].to_dict()
    elif isinstance(df2, dict):
        map_dict = df2
    else:
        raise ValueError("df2 phải là DataFrame hoặc dict")

    df1[new_column_name] = df1[column_df1].map(map_dict)
    df1[new_column_name] = df1[new_column_name].apply(
        lambda x: format_array(x) if isinstance(x, (list, np.ndarray)) else x
    )
    return df1
    
def encode_and_save_csv(df, output_path, columns_to_encode):
    label_encoders = {}
    for column in columns_to_encode:
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])
    df.to_csv(output_path, index=False)
    return label_encoders

def setup_path():
    allreviews_path = "feature/allFeatureReview_"
    reviewer_path = "feature/reviewer_feature_"
    item_path = "feature/item_feature_"
    udeep_path = "feature/u_deep_"
    ideep_path = "feature/i_deep_"
    tranformed_udeep_path = "feature/transformed_udeep_"
    tranformed_ideep_path = "feature/transformed_ideep_"
    final_data_path = "data/final_data_feature_"
    svd_path = "chkpt/svd_"
    checkpoint_path = 'chkpt/fm_checkpoint_'
    sparse_matrix_path = 'chkpt/encoded_features_'
    
    return allreviews_path, reviewer_path, item_path, udeep_path, ideep_path, tranformed_udeep_path, tranformed_ideep_path, final_data_path, svd_path, checkpoint_path, sparse_matrix_path

if __name__ == "__main__":
    import nltk
    nltk.download('averaged_perceptron_tagger')