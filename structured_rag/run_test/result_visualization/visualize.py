import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pydantic import BaseModel
from enum import Enum
from typing import List

from models import PromptWithResponse, PromptingMethod, Experiment

def load_experiments(directory: str) -> pd.DataFrame:
    experiments = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), 'r') as f:
                data = json.load(f)
                experiment = Experiment(**data)
                experiments.append({
                    'test_name': experiment.test_name,
                    'model_name': experiment.model_name,
                    'prompting_method': experiment.prompting_method,
                    'num_successes': experiment.num_successes,
                    'num_attempts': experiment.num_attempts,
                    'success_rate': experiment.success_rate,
                    'total_time': experiment.total_time,
                    'avg_response_time': experiment.total_time / experiment.num_attempts
                })
    return pd.DataFrame(experiments)

def barplot_success_rates(df: pd.DataFrame):
    """
    This function plots the success rates of the models, averaged over the two prompting methods.
    """
    # compute the average success rate over the two prompting methods
    df_avg = df.groupby(['model_name', 'prompting_method'])['success_rate'].mean().reset_index()
    plt.figure(figsize=(12, 6))
    sns.barplot(x='model_name', y='success_rate', data=df_avg)
    plt.title('Success Rates by Model and Prompting Method')
    plt.xlabel('Model Name')
    plt.ylabel('Success Rate')
    plt.savefig('success_rates.png')
    plt.close()

def barplot_success_rates_per_test(df: pd.DataFrame):
    """
    This function plots the success rates of the models for each test.
    """
    df_avg = df.groupby(['model_name', 'test_name'])['success_rate'].mean().reset_index()
    plt.figure(figsize=(15, 8))
    sns.barplot(x='test_name', y='success_rate', hue='model_name', data=df_avg)
    plt.title('Success Rates by Model and Test')
    plt.xlabel('Test Name')
    plt.ylabel('Success Rate')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Model Name', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('success_rates_per_test.png')
    plt.close()

def plot_success_rate_heatmap(df: pd.DataFrame, models: List[str]):
    # Filter the dataframe for the specified models
    df_filtered = df[df['model_name'].isin(models)]
    
    # Create a new column combining model_name and prompting_method
    df_filtered['model_method'] = df_filtered['model_name'].apply(lambda x: 'claude-3-5-sonnet' if x == 'claude-3-5-sonnet-20240620' else x) + '_' + df_filtered['prompting_method']
    
    # Pivot the table
    pivot_df = df_filtered.pivot_table(values='success_rate', index='test_name', columns='model_method', aggfunc='mean')
    
    # Reorder columns to group by model
    column_order = [f"{model if model != 'claude-3-5-sonnet-20240620' else 'claude-3-5-sonnet'}_{method}" for model in models for method in ['dspy', 'fstring']]
    pivot_df = pivot_df[column_order]
    
    plt.figure(figsize=(12, 10))
    # Create a custom colormap from red (0%) to green (100%)
    cmap = mcolors.LinearSegmentedColormap.from_list("", ["red", "yellow", "green"])
    sns.heatmap(pivot_df, annot=True, cmap=cmap, fmt='.2f', vmin=0, vmax=1)
    plt.title('Success Rate Heatmap')
    plt.xlabel('Model and Prompting Method')
    plt.ylabel('Test Name')
    plt.xticks(rotation=45, ha='right')
    
    # Rename 'claude-3-5-sonnet-20240620' to 'claude-3-5-sonnet' in x-axis labels
    x_labels = [label.get_text().replace('claude-3-5-sonnet-20240620', 'claude-3-5-sonnet') for label in plt.gca().get_xticklabels()]
    # replace 'dspy' with 'FF'
    x_labels = [label.get_text().replace('dspy', 'FF') for label in plt.gca().get_xticklabels()]
    plt.gca().set_xticklabels(x_labels)
    
    plt.tight_layout()
    plt.savefig('success_rate_heatmap.png')
    plt.close()

def boxplot_success_rate_per_task(df: pd.DataFrame):
    """
    This function plots the success rates for each test, averaged across all models.
    """
    plt.figure(figsize=(12, 10))
    
    # Create the box plot
    sns.boxplot(x='success_rate', y='test_name', data=df, orient='h', color='lightblue', width=0.5)
    
    # Customize the plot
    plt.title('Success Rates by Test', fontsize=16)
    plt.xlabel('Success Rate', fontsize=12)
    plt.ylabel('Test Name', fontsize=12)
    plt.xlim(0, 1)  # Set x-axis limits from 0 to 1 for success rate
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Improve readability
    plt.tight_layout()
    plt.savefig('boxplot_success_rates_per_task.png')
    plt.close()

def boxplot_success_rate_per_model(df: pd.DataFrame):
    """
    This function plots the success rates for each model, averaged across all tests.
    """
    plt.figure(figsize=(12, 10))
    
    # Create the box plot
    sns.boxplot(x='model_name', y='success_rate', data=df, color='lightblue', width=0.5)
    
    # Customize the plot
    plt.title('Success Rates by Model', fontsize=16)
    plt.xlabel('Model Name', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.ylim(0, 1)  # Set y-axis limits from 0 to 1 for success rate
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Improve readability
    plt.tight_layout()
    plt.savefig('boxplot_success_rates_per_model.png')
    plt.close()

def visualize_experiments(df: pd.DataFrame):
    # Set the style for all plots
    plt.style.use('ggplot')

    barplot_success_rates(df)
    barplot_success_rates_per_test(df)
    plot_success_rate_heatmap(df, models=["claude-3-5-sonnet-20240620", "llama3:instruct"])
    boxplot_success_rate_per_task(df)
    boxplot_success_rate_per_model(df)
    
if __name__ == "__main__":
    # Load experiments from the 'experiments' directory
    df = load_experiments('experimental-results-9-11-24')
    
    # Generate visualizations
    visualize_experiments(df)
    
    print("Visualizations have been generated and saved as PNG files.")