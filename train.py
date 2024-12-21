import torch
import torch.nn as nn
import torch.optim as optim
import ast
import numpy as np
import tqdm
from sklearn.linear_model import LinearRegression
from helper.general_functions import create_and_write_csv, word_segment
from combine_review_rating import Calculate_Deep, mergeReview_Rating
from sklearn.metrics import accuracy_score, mean_absolute_error
from review_processing.merge_senmatic_review import extract_features, initialize_features
from helper.utils import encode_and_save_csv, map_and_add_column, setup_path
from rating_processing.svd import initialize_svd
from rating_processing.factorization_machine import run

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

def reprocess_input(data):
    rating = torch.tensor([float(x) for x in data['overall']], dtype=torch.float32)
    item_bias = torch.tensor([float(x) for x in data['item_bias']], dtype=torch.float32)
    user_bias = torch.tensor([float(x) for x in data['user_bias']], dtype=torch.float32)

    user_feature = []
    for item in data['Udeep']:
        if isinstance(item, str):
            user_feature.append(torch.tensor(ast.literal_eval(item), dtype=torch.float32))
        elif isinstance(item, np.ndarray):
            user_feature.append(torch.tensor(item, dtype=torch.float32))
        else:
            user_feature.append(item.float())
    
    item_feature = []
    for item in data['Ideep']:
        if isinstance(item, str):
            item_feature.append(torch.tensor(ast.literal_eval(item), dtype=torch.float32))
        elif isinstance(item, np.ndarray):
            item_feature.append(torch.tensor(item, dtype=torch.float32))
        else:
            item_feature.append(item.float())
    
    user_feature = torch.stack(user_feature)
    item_feature = torch.stack(item_feature)
    
    return rating, user_feature, item_feature, item_bias, user_bias

def calculate_rmse(y_true, y_pred):
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    squared_errors = (y_true_np - y_pred_np) ** 2
    mean_squared_error = np.mean(squared_errors)
    rmse = np.sqrt(mean_squared_error)
    return rmse

# Define the model
class FullyConnectedModel(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(FullyConnectedModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc = nn.Linear(input_dim, output_dim, bias=True)
        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self, user_features, item_features, item_bias, user_bias):
        interaction = user_features * item_features
        interaction_sum = interaction.sum(dim=1)
        if(len(interaction_sum.size()) != self.input_dim):
            self.fc = nn.Linear(len(interaction_sum), len(interaction_sum), bias=True)
            
        # Multiply by weights
        prediction = self.fc(interaction_sum.to(dtype=torch.float32))
        prediction += self.global_bias + item_bias.squeeze()  + user_bias.squeeze() 
        return prediction.squeeze()

    
def train_deepbert(train_data_loader, valid_data_loader, num_factors, batch_size, epochs, method_name, log_interval=100):
    print("=================== Training DeepCGSR model ============================")
    model = FullyConnectedModel(input_dim=batch_size, output_dim=num_factors)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.09)
    early_stopper = EarlyStopper(num_trials=5, save_path=f'chkpt/{method_name}.pt')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_data_loader):
            try:
                rating, user_feature, item_feature, item_bias, user_bias = reprocess_input({
                    'reviewerID': batch[0],
                    'itemID': batch[1],
                    'overall': batch[2],
                    'Udeep': batch[3],
                    'Ideep': batch[4],
                    'item_bias': batch[5],
                    'user_bias': batch[6],
                })
                
                predictions = model(user_feature, item_feature, item_bias, user_bias)
                print("predictions: ", predictions)
                # predictions = torch.clamp(predictions, min=1.0, max=5.0)
                loss = criterion(predictions, rating)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
                if (batch_idx + 1) % log_interval == 0:
                    print(f"Train Epoch: {epoch+1} [{batch_idx * len(batch[0])}/{len(train_data_loader.dataset)} "
                          f"({100. * batch_idx / len(train_data_loader):.0f}%)]\tLoss: {loss.item():.6f}")
                    
            except Exception as e:
                print("Error: ", e)

        auc = test(model, valid_data_loader) 
        # print(f"Validation AUC: {auc}")
        
        if not early_stopper.is_continuable(model, auc):
            print(f'validation: best auc: {early_stopper.best_accuracy}')
            break

    return model

def test(model, data_loader):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            data = {
                'reviewerID': batch[0],
                'itemID': batch[1],
                'overall': batch[2],
                'Udeep': batch[3],
                'Ideep': batch[4],
                'item_bias': batch[5],
                'user_bias': batch[6],
            }
            
            target, udeep, ideep, item_bias, user_bias = reprocess_input(data)
            udeep = torch.tensor(udeep, dtype=torch.float32) if isinstance(udeep, list) else udeep
            ideep = torch.tensor(ideep, dtype=torch.float32) if isinstance(ideep, list) else ideep

            y = model(udeep, ideep, item_bias, user_bias)
            
            targets.extend(target)
            predicts.extend([round(float(pred)) for pred in y.flatten().cpu().numpy()])

    new_targets = [-1 if i < 4 else 1 for i in targets]
    new_predicts = [-1 if i < 4 else 1 for i in predicts]

    accuracy = accuracy_score(new_targets, new_predicts)
    print("Accuracy: ", accuracy)
    return accuracy

def test_rsme(model, data_loader):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            data = {
                'reviewerID': batch[0],
                'itemID': batch[1],
                'overall': batch[2],
                'Udeep': batch[3],
                'Ideep': batch[4],
                'item_bias': batch[5],
                'user_bias': batch[6],
            }
            
            target, udeep, ideep, item_bias, user_bias = reprocess_input(data)
            y = model(udeep, ideep, item_bias, user_bias)
            targets.extend(target)
            predicts.extend([float(pred) for pred in y.flatten().cpu().numpy()])

    new_targer = []
    new_predict = []
    new_targer = targets
    new_predict = new_predict

    print("rsme raw: ", calculate_rmse(targets, predicts))
    mae_value = mean_absolute_error(targets, predicts)
    print("MAE: ", mae_value)
    return calculate_rmse(targets, predicts), mae_value

def calulate_user_item_bias(allFeatureReviews):
    print("allFeatureReviews: ", allFeatureReviews)
    allFeatureReviews['Ideep'] = allFeatureReviews['Ideep'].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float64))
    allFeatureReviews['Udeep'] = allFeatureReviews['Udeep'].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float64))
    
    item_features = np.vstack(allFeatureReviews['Ideep'].tolist())
    user_features = np.vstack(allFeatureReviews['Udeep'].tolist())

    ratings = np.array(allFeatureReviews['overall'].tolist()).astype(np.float64)
    ratings = np.nan_to_num(ratings, nan=0.0)

    item_bias = calculate_bias(item_features, ratings)
    user_bias = calculate_bias(user_features, ratings)    
    
    return item_bias, user_bias

def calculate_bias(feature_vectors, ratings):    
    model = LinearRegression()
    model.fit(feature_vectors, ratings)
    predicted_ratings = model.predict(feature_vectors)
    bias = ratings - predicted_ratings
    return bias



# DeepBERT
def DeepBERT(dataset_df, num_factors, num_words, filename):

    train_data_list = dataset_df["filteredReviewText"].tolist() 
    allreviews_path, reviewer_path, item_path, _, _, _, _, final_data_path, svd_path, checkpoint_path, sparse_matrix_path = setup_path()   
    split_data = []

    for i in train_data_list:
        split_data.append(word_segment(i))


    allFeatureReviews = extract_features(dataset_df, split_data, num_factors, num_words, filename)
    reviewer_feature_dict, item_feature_dict = initialize_features(filename, num_factors)
    
    svd = initialize_svd(allreviews_path + filename + ".csv", num_factors, svd_path + filename +'.pt')
    z_item = mergeReview_Rating(item_path + filename +".csv", "z_item_" + filename, svd, reviewer_feature_dict, item_feature_dict, "item")
    z_review = mergeReview_Rating(reviewer_path + filename +".csv", "z_reviewer_" + filename, svd, reviewer_feature_dict, item_feature_dict, "reviewer")
    
    v_reviewer_list = []
    v_item_list = []
    fm = run(allreviews_path + filename +".csv", num_factors * 2, checkpoint_path + filename +'.pkl', sparse_matrix_path + filename +'.npz')
    for name in z_review.items():
        v_reviewer_list.append(fm.get_embedding('reviewerID_' + name[0]))

    for name in z_item.items():
        v_item_list.append(fm.get_embedding('itemID_' + name[0]))
        
        
    print("================")
    u_deep = {}
    i_deep = {}

    for (z_name, z_value), v_value  in zip(z_review.items(), v_reviewer_list):
        u_deep[z_name] = Calculate_Deep(z_value, v_value)

    for (z_name, z_value), v_value in zip(z_item.items(), v_item_list):
        i_deep[z_name] = Calculate_Deep(z_value, v_value)
        
    create_and_write_csv("u_deep_" + filename, u_deep)
    create_and_write_csv("i_deep_" + filename, i_deep)

    allFeatureReviews = allFeatureReviews[['reviewerID', 'itemID', 'overall']]
    allFeatureReviews = map_and_add_column(allFeatureReviews, u_deep, 'reviewerID', 'Key', 'Array', 'Udeep')
    allFeatureReviews = map_and_add_column(allFeatureReviews, i_deep, 'itemID', 'Key', 'Array', 'Ideep')
 
    item_bias, user_bias = calulate_user_item_bias(allFeatureReviews)
    allFeatureReviews['item_bias'] = item_bias
    allFeatureReviews['user_bias'] = user_bias
    
    encode_and_save_csv(allFeatureReviews, final_data_path + "DeepBERT" + "_" + filename +".csv", ['reviewerID', 'itemID'])
    


