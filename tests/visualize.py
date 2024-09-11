import os
import json
import pandas as pd
import matplotlib.pyplot as plt
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

def plot_success_rate_vs_response_time(df: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='avg_response_time', y='success_rate', hue='model_name', style='prompting_method', data=df)
    plt.title('Success Rate vs. Average Response Time')
    plt.xlabel('Average Response Time (ms)')
    plt.ylabel('Success Rate')
    plt.savefig('success_rate_vs_response_time.png')
    plt.close()

def plot_response_time_distribution(df: pd.DataFrame):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='model_name', y='avg_response_time', hue='prompting_method', data=df)
    plt.title('Distribution of Response Times by Model and Prompting Method')
    plt.xlabel('Model Name')
    plt.ylabel('Average Response Time (ms)')
    plt.savefig('response_time_distribution.png')
    plt.close()

def plot_success_rate_heatmap(df: pd.DataFrame):
    pivot_df = df.pivot_table(values='success_rate', index='model_name', columns='prompting_method', aggfunc='mean')
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='.2f')
    plt.title('Success Rate Heatmap')
    plt.savefig('success_rate_heatmap.png')
    plt.close()

def visualize_experiments(df: pd.DataFrame):
    # Set the style for all plots
    plt.style.use('ggplot')

    barplot_success_rates(df)
    #plot_success_rate_vs_response_time(df)
    #plot_response_time_distribution(df)
    #plot_success_rate_heatmap(df)

if __name__ == "__main__":
    # Load experiments from the 'experiments' directory
    df = load_experiments('experimental-results-9-11-24')
    
    # Generate visualizations
    visualize_experiments(df)
    
    print("Visualizations have been generated and saved as PNG files.")