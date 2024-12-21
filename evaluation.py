import json
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import train_test_split
from helper.general_functions import save_to_excel
from helper.utils import csv_to_dataloader, read_data
from train import DeepBERT, test, test_rsme, train_deepbert


# from model.DeepCGSR.train import train

class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save(model, self.save_path) # best model
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False

# Hàm để tạo DataLoader cho các tập train, validation, test
def create_dataloaders(json_file, batch_size=32, train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1):
    dataset = read_data(json_file)
    total_size = len(dataset)
    
    train_size = int(train_ratio * total_size)
    valid_size = int(valid_ratio * total_size)
    test_size = total_size - train_size - valid_size
    
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader

# Chia dữ liệu thành các DataFrame (train, validation, test)
def create_dataframes(json_file, train_ratio=0.7, valid_ratio=0.1, test_ratio=0.2):
    data = read_data(json_file)
    df = pd.DataFrame(data, columns=['reviewerID', 'asin', 'overall', 'overall_new', 'reviewText', 'filteredReviewText'])
    
    train_df, temp_df = train_test_split(df, train_size=train_ratio, random_state=42)
    valid_ratio_temp = valid_ratio / (valid_ratio + test_ratio)
    valid_df, test_df = train_test_split(temp_df, train_size=valid_ratio_temp, random_state=42)
    
    return df, train_df, valid_df, test_df

# Ví dụ sử dụng
dataset_name = ["Small_All_Beauty_5"]
batch_size = 32
num_epochs = 100
list_factors = [40]
num_words = 200
is_switch_data = True
rsme_MFFR = 0
mae_MFFR = 0
f1_MFFR = 0
for dataset in dataset_name:
    for num_factors in list_factors:
        json_file = "data/" + dataset + ".json"
        all_df, train_df, valid_df, test_df = create_dataframes(json_file)

    #region DeepCGSR 
        method_name = ["DeepBERT"]
        for method in method_name:
            print("Method: ", method)
            
            DeepBERT(train_df, num_factors, num_words, "train")
            DeepBERT(valid_df, num_factors, num_words, "vaild")
            DeepBERT(test_df, num_factors, num_words, "test")

            final_feature_train_path = "data/final_data_feature_" + method + "_train.csv"
            final_feature_valid_path = "data/final_data_feature_" + method + "_train.csv"
            final_feature_test_path = "data/final_data_feature_" + method + "_test.csv"

            train_data_loader = csv_to_dataloader(final_feature_train_path, batch_size)
            valid_data_loader = csv_to_dataloader(final_feature_valid_path, batch_size)
            test_data_loader = csv_to_dataloader(final_feature_test_path, batch_size)
            
            model_deep = train_deepbert(train_data_loader, valid_data_loader, num_factors, batch_size, num_epochs, method, log_interval=100)
            auc_test = test(model_deep, test_data_loader)
            rsme_test, mae_test = test_rsme(model_deep, test_data_loader)
            DeepCGSR_results = [auc_test, rsme_test, mae_test]
            
            save_to_excel([DeepCGSR_results], ['AUC', 'RSME Test', 'MAE Test'], "results/"+ method + "_" + dataset + "_factors" + str(num_factors) + ".xlsx")
        #endregion



