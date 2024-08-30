from joblib import load
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

def test_base_model(test_data_processed, test_labels):
    # Load the baseline model
    baseline_model = load('src/models/serialized/baseline_model.pkl')
    
    # Predict test data using the baseline model
    baseline_test_predictions = baseline_model.predict(test_data_processed)

    # Evaluate intitial results of training predictions
    accuracy = accuracy_score(test_labels, baseline_test_predictions)
    precision = precision_score(test_labels, baseline_test_predictions)
    recall = recall_score(test_labels, baseline_test_predictions)
    f1 = f1_score(test_labels, baseline_test_predictions)

    cm = confusion_matrix(test_labels, baseline_test_predictions)

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
    plt.savefig('reports/figures/testing/baseline_test_confusion_matrix.png')  # Save the plot as a file
    plt.close()  # Close the plot to free up memory

    return metrics_dict


if __name__ == '__main__':
    test_data_processed = pd.read_csv('data/processed/test_data_processed.csv')
    test_labels = pd.read_csv('data/test_data/test_labels.csv')
    baseline_test_results = test_base_model(test_data_processed, test_labels)
    print(baseline_test_results)