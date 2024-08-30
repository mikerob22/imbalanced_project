import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from joblib import dump

def train_baseline_model(train_data, train_labels):

    """
    Train a baseline logistic regression model and evaluate its performance.

    Parameters:
    train_data (DataFrame): The training features.
    train_labels (Series): The training labels.

    Returns:
    dict: A dictionary containing evaluation metrics.
    """

    # Apply Multiple Logistic Regression
    baseline_model = LogisticRegression()
    baseline_model.fit(train_data, train_labels)

    # Predict training data
    baseline_model_preds = baseline_model.predict(train_data)

    # Evaluate intitial results of training predictions
    accuracy = accuracy_score(train_labels, baseline_model_preds)
    precision = precision_score(train_labels, baseline_model_preds)
    recall = recall_score(train_labels, baseline_model_preds)
    f1 = f1_score(train_labels, baseline_model_preds)

    cm = confusion_matrix(train_labels, baseline_model_preds)

    # Create a dictionary with all the metrics
    metrics_dict = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Confusion Matrix': cm.tolist()  # Convert numpy array to list for better readability
    }

    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=baseline_model.classes_)
    disp.plot()
    plt.savefig('reports/figures/training/baseline_model_confusion_matrix.png')  # Save the plot as a file
    plt.close()  # Close the plot to free up memory

    # Serialize baseline model to use for testing
    dump(baseline_model, 'src/models/serialized/baseline_model.pkl')

    return metrics_dict


if __name__ == "__main__":
    train_data_processed = pd.read_csv('data/processed/train_data_processed.csv')
    train_labels = pd.read_csv('data/train_data/train_labels.csv')
    baseline_model_results = train_baseline_model(train_data_processed, train_labels)
    print(baseline_model_results)
