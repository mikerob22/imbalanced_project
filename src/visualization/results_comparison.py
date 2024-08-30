import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import plot

base_test_results = {
    'Accuracy': 0.775, 
    'Precision': 0.5701357466063348, 
    'Recall': 0.17847025495750707, 
    'F1-Score': 0.27184466019417475, 
    'Confusion Matrix': [[2199, 95], [580, 126]]
    }


smote_test_results = {
    'Accuracy': 0.7633333333333333, 
    'Precision': 0.49821109123434704, 
    'Recall': 0.7889518413597734, 
    'F1-Score': 0.6107456140350878, 
    'Confusion Matrix': [[1733, 561], [149, 557]]
    }


# FUNCTION TO CREATE COMPARISON BAR CHART WITH MATPLOTLIB
def visualize_metrics():
    # Metrics to plot
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    base_values = [base_test_results[metric] for metric in metrics]
    smote_values = [smote_test_results[metric] for metric in metrics]

    # X-axis locations
    x = np.arange(len(metrics))

    # Bar width
    width = 0.35

    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width/2, base_values, width, label='Base Model')
    bars2 = ax.bar(x + width/2, smote_values, width, label='SMOTE Model')

    # Labels and title
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.set_title('Comparison of Test Results')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    # Adding value labels on top of the bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    add_labels(bars1)
    add_labels(bars2)

    plt.tight_layout()
    plt.show()


# FUNCTION TO CREATE COMPARISON BAR CHART WITH SEABORN
def visualize_metrics_seaborn():
    # Prepare the data
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    base_values = [base_test_results[metric] for metric in metrics]
    smote_values = [smote_test_results[metric] for metric in metrics]

    # Create a DataFrame for Seaborn
    df = pd.DataFrame({
        'Metrics': metrics * 2,
        'Values': base_values + smote_values,
        'Model': ['Base Model'] * len(metrics) + ['SMOTE Model'] * len(metrics)
    })

    # Create the barplot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Metrics', y='Values', hue='Model', data=df)
    plt.title('Comparison of Test Results')
    plt.ylabel('Values')
    plt.xlabel('Metrics')

    # Display the values on the bars
    for container in plt.gca().containers:
        plt.gca().bar_label(container, fmt='%.2f')

    plt.show()


# FUNCTION TO CREATE COMPARISON BAR CHART WITH PLOTLY
def visualize_metrics_plotly():
    # Prepare the data
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    base_values = [base_test_results[metric] for metric in metrics]
    smote_values = [smote_test_results[metric] for metric in metrics]

    # Create traces
    trace1 = go.Bar(
        x=metrics,
        y=base_values,
        name='Base Model',
        text=[f'{val:.2f}' for val in base_values],
        textposition='auto'
    )
    trace2 = go.Bar(
        x=metrics,
        y=smote_values,
        name='SMOTE Model',
        text=[f'{val:.2f}' for val in smote_values],
        textposition='auto'
    )

    # Create the layout
    layout = go.Layout(
        title='Comparison of Test Results',
        xaxis=dict(title='Metrics'),
        yaxis=dict(title='Values'),
        barmode='group'
    )

    # Create the figure
    fig = go.Figure(data=[trace1, trace2], layout=layout)

    # Show the figure
    plot(fig)

    # Convert the plot to HTML string
    plot_div = plot(fig, output_type='div', include_plotlyjs=False)

    return plot_div




# TODO: Figure out why the plot is being saved in workspace dir as 'temp-plot.html' instead of defined dir

def plotly_plot_to_html():
    # Saving plot as HTML string
    plot_html = visualize_metrics_plotly()
    with open("reports/figures/testing/plot_div.html", "w") as file:
        file.write(plot_html)



if __name__ == "__main__":
    visualize_metrics()
    visualize_metrics_seaborn()
    visualize_metrics_plotly()