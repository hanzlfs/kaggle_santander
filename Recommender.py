
"""Santander Product Recommmendation
This is my part of the code that produced the best submission to the
Santander Product Recommendation. See README.md for more details.
Author: Mu Tian <kevinmtian@gmail.com> Zhonglin Han <hanzhonglin1990@gmail.com >
"""

from helper.config import get_config, get_model
from helper.ml import train_model, create_prediction, create_submission
from helper.data_process import get_processed_data
from helper.feature_generation import create_data

def main():
    """
    The best submission was a single LightGBM model. 
    We provide flexibility for various model and feature configuration. 
    See README.md for more details.
    """    

    #Config
    CONFIG_DATA, CONFIG_FEATURES, CONFIG_PARAMS = get_config()
    CLF = get_model(model = "LGBM", param = CONFIG_PARAMS):

    #Load Data
    train_current, train_previous, test_current, test_previous, mapping_dict = \
        get_processed_data(CONFIG_DATA["path_current_train"], CONFIG_DATA["path_previous_train"], \
            CONFIG_DATA["path_current_test"], CONFIG_DATA["path_previous_test"])

    #Create Features
    train_label, train_data, train_previous_products, train_user_ids, train_new_products = \
        create_data(feature_msg = CONFIG_FEATURES, translation_dict = mapping_dict, \
                    train_current = train_current, train_previous = train_previous, \
                    test_current = test_current, test_previous = test_previous, \
                    return_labels = True, return_prev_prod = True, return_user_id = True, return_new_products = True, \
                    MONTHS = CONFIG_FEATURES['train_month'], is_train = True)

    test_label, test_data, test_previous_products, test_user_ids, test_new_products = \
        create_data(feature_msg = CONFIG_FEATURES, translation_dict = mapping_dict, \
                    train_current = train_current, train_previous = train_previous, \
                    test_current = test_current, test_previous = test_previous, \
                    return_labels = True, return_prev_prod = True, return_user_id = True, return_new_products = True, \
                    MONTHS = CONFIG_FEATURES['test_month'], is_train = False)
    
    # Train
    CLF, unique_labels = ml.train_model(y = train_label, X = train_data, model = CLF)

    # Test
    predictions = ml.create_prediction(model = CLF, X = test_data, previous_products = test_previous_products, \
        unq_lb = unique_labels) # generate prediction
    create_submission(predictions = predictions, test_id = test_user_ids, \
        filename = CONFIG_DATA["filename"])

if __name__ == '__main__':

    main()






