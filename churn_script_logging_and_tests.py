'''
Test script for churn_library.py
'''

import logging
import time
import churn_library as cl

logging.basicConfig(
    filename='./logs/churn_libary.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s = %(message)s'
)


def test_import(import_data):
    '''
     test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        data_frame = import_data("./data/bank_data.csv")
        logging.info("Testing import_data successful: File loaded correctly")
    except FileNotFoundError as err:
        logging.error("Testing import_data exception: The file wasn't found")
        raise err

    try:
        assert data_frame.shape[0] > 0
        assert data_frame.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data exception: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    try:
        data_frame = cl.import_data("./data/bank_data.csv")
        perform_eda(data_frame)
        logging.info("Testing perform_eda successful: EDA performed correctly")
    except AssertionError as err:
        logging.error("Testing perform_eda exception: Invalid dataframe")
        raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    try:
        data_frame = cl.import_data("./data/bank_data.csv")
        X = encoder_helper(data_frame, cl.category_list)
        logging.info(
            "Testing encoder_helper successful: Encoder performed correctly")
    except AssertionError as err:
        logging.error("Testing encoder_helper exception: XXX")
        raise err

    try:
        assert X.shape[0] > 0
        assert X.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing encoder_helper exception: XXXX")
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    try:
        data_frame = cl.import_data("./data/bank_data.csv")
        X_train, X_test, y_train, y_test = perform_feature_engineering(
            data_frame, 'Churn')
        logging.info(
            "Testing perform_feature_engineering successful: data split")
    except AssertionError as err:
        logging.error("Testing perform_feature_engineering exception: XXX")
        raise err

    try:
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert X_train.shape[1] > 0
        assert X_test.shape[1] > 0
        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]

    except AssertionError as err:
        logging.error("Testing perform_feature_engineering exception: XXXX")
        raise err


def test_train_models(train_models):
    '''
    test train_models
    '''
    data_frame = cl.import_data(r"./data/bank_data.csv")
    X_train, X_test, y_train, y_test = cl.perform_feature_engineering(
        data_frame, 'Churn')
    try:
        train_models(X_train, X_test, y_train, y_test)
        logging.info("Testing train_models successful: Models trained")
    except AssertionError as err:
        logging.error("Testing train_models exception: Invalid size of data")
        raise err
    except AttributeError as err:
        logging.error(
            "Testing train_models exception: Input is not a data frame")
        raise err

    try:
        model_1 = open('./models/rfc_model.pkl')
        model_1.close()
        model_2 = open('./models/logistic_model.pkl')
        model_2.close()
    except FileNotFoundError:
        logging.error("Testing train_models exception: Model not saved")
    try:
        assert X_train.shape[0] > 0

    except AssertionError as err:
        logging.error("Testing train_models exception: XXXX")
        raise err


if __name__ == "__main__":
    print("start test")
    start_time = time.time()
    test_import(cl.import_data)
    print("--- import_data took %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    test_eda(cl.perform_eda)
    print("--- perform_eda took %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    test_encoder_helper(cl.encoder_helper)
    print(
        "--- encoder_helper took %s seconds ---" %
        (time.time() - start_time))

    start_time = time.time()
    test_perform_feature_engineering(cl.perform_feature_engineering)
    print(
        "--- perform_feature_engineering took %s seconds ---" %
        (time.time() - start_time))

    start_time = time.time()
    test_train_models(cl.train_models)
    print("--- train_models took %s seconds ---" % (time.time() - start_time))
    print("finished")
