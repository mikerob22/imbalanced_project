import pandas as pd
import numpy as np
from src.features.feature_engineering import feature_engineering
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split



# Load and split data into train set and test set
# use preprocessing function on raw train set and then feed through to models
# store training_data_processed and training_labels_processed in data/processed folder
# use processed training data instead of preprocessing the whole dataset first


def load_and_split_data(filepath):

    """
    Load and split data into training and testing datasets.

    Parameters:
    filepath (str): The path to the CSV file containing the dataset.

    Returns:
    tuple: train_data, train_labels, test_data, test_labels
    """

    data = pd.read_csv(filepath)

    train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
    
    ### HANDLE TRAIN SET ### 
    train_data = train_set.drop('left', axis=1)
    train_labels = pd.Series(train_set['left'])

    train_data.to_csv('data/train_data/train_data.csv', index=False)
    train_labels.to_csv('data/train_data/train_labels.csv', index=False)

    ### HANDLE TEST SET ### 
    test_data = test_set.drop('left', axis=1)
    test_labels = pd.Series(test_set['left'])

    test_data.to_csv('data/test_data/test_data.csv', index=False)
    test_labels.to_csv('data/test_data/test_labels.csv', index=False)

    return train_data, train_labels, test_data, test_labels


def preprocess_data(data):
    # Implement preprocessing steps specific to baseline model

    ### SEPARATE BINARY FEATURES ALREADY ENCODED ###
    binary_features = ['Work_accident', 'promotion_last_5years']
    binary_data = data[binary_features].copy()

    data_non_binary = data.drop(columns=binary_features)

    ### ENCODING CATEGORICAL DATA ###
    categorical_features = data_non_binary.select_dtypes(exclude=np.number).columns
    data_categotical = data_non_binary[categorical_features]

    cat_encoder = OneHotEncoder(sparse_output=False)
    cat_onehot = cat_encoder.fit_transform(data_categotical)
    
    categorical_feature_names = cat_encoder.get_feature_names_out(categorical_features)

    ### FEATURE ENGINEERING ###
    feature_engineered = feature_engineering(data_non_binary)


    ### SCALING NUMERICAL DATA ###
    numerical_features = feature_engineered.select_dtypes(include=np.number).columns
    data_numerical = feature_engineered[numerical_features]

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_numerical)


    ### Combine Categorical and Numerical Features ###
    processed_array = np.hstack([cat_onehot, data_scaled, binary_data.values])
    processed_features = pd.Series(np.hstack([categorical_feature_names, numerical_features, binary_features]))


    ### Create Processed DataFrame ###
    processed_df = pd.DataFrame(processed_array, columns=processed_features)

    return processed_df, processed_features


if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels = load_and_split_data('data/raw/turnover.csv')
    train_data_processed, train_features_processed = preprocess_data(train_data)
    test_data_processed, test_features_processed = preprocess_data(test_data)
    train_data_processed.to_csv('data/processed/train_data_processed.csv')
    train_features_processed.to_csv('data/processed/train_features_processed.csv', index=False)
    test_data_processed.to_csv('data/processed/test_data_processed.csv')
    test_features_processed.to_csv('data/processed/test_features_processed.csv', index=False)