
import numpy as np
from helper.general_functions import create_and_write_csv, read_csv_file


#============================ Calulate U/I deep ===============================

def Calculate_Deep(v, z):
    v_z = v * z
    v2_z2 = (v**2) * (z**2)
    result = (1 / 2) * ((v_z)**2 - v2_z2)
    return result


def mergeReview_Rating(path, filename, svd, reviewer_feature_dict, item_feature_dict, getEmbedding):
    reviewerID,_ = read_csv_file(path)
    feature_dict = {}
    review_feature_list = []
    rating_feature_list = []
    for id in reviewerID:
        if getEmbedding == "reviewer":
            A = reviewer_feature_dict[id]
            B = svd.get_user_embedding(id)
        else:
            A = item_feature_dict[id]
            B = svd.get_item_embedding(id)

        z = np.concatenate((np.array(A), np.array(B)))
        feature_dict[id] = z
        review_feature_list.append(A)
        rating_feature_list.append(B)
    create_and_write_csv(filename, feature_dict)
    return feature_dict




