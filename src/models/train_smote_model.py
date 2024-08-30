import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from joblib import dump

def smote_resample():
    train_data_processed = pd.read_csv('data/processed/train_data_processed.csv')
    train_labels = pd.read_csv('data/train_data/train_labels.csv').squeeze()

    # Apply SMOTE to the preprocessed data, it automatically will apply a balance of 50/50 for target variable classes
    smote = SMOTE(random_state=42)
    train_data_resampled, train_labels_resampled = smote.fit_resample(train_data_processed, train_labels )

    return train_data_resampled, train_labels_resampled


def train_smote_model(train_data_resampled, train_labels_resampled):

    """
    Train a logistic regression model with SMOTE applied and evaluate its performance.

    Parameters:
    train_data_resampled (DataFrame): The training features resampled with SMOTE.
    train_labels_resampled (Series): The training labels resampled with SMOTE.

    Returns:
    dict: A dictionary containing evaluation metrics.
    """

    # Apply Multiple Logistic Regression
    smote_model = LogisticRegression()
    smote_model.fit(train_data_resampled, train_labels_resampled)

    # Predict training data
    smote_model_preds = smote_model.predict(train_data_resampled)

    # Evaluate intitial results of training predictions
    accuracy = accuracy_score(train_labels_resampled, smote_model_preds)
    precision = precision_score(train_labels_resampled, smote_model_preds)
    recall = recall_score(train_labels_resampled, smote_model_preds)
    f1 = f1_score(train_labels_resampled, smote_model_preds)

    cm = confusion_matrix(train_labels_resampled, smote_model_preds)

    # Create a dictionary with all the metrics
    metrics_dict = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Confusion Matrix': cm.tolist()  # Convert numpy array to list for better readability
    }

    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=smote_model.classes_)
    disp.plot()
    plt.savefig('reports/figures/training/smote_model_confusion_matrix.png')  # Save the plot as a file
    plt.close()  # Close the plot

    # Serialize baseline model to use for testing
    dump(smote_model, 'src/models/serialized/smote_model.pkl')

    return metrics_dict


if __name__ == "__main__":
    train_data_resampled, train_labels_resampled = smote_resample()
    smote_model_results = train_smote_model(train_data_resampled, train_labels_resampled)
    print(smote_model_results)