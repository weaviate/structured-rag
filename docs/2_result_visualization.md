# Result Visualization

To visualize the results of your experiments, follow these steps:

1. Aggregate results from each trial into a single file:
   ```
   python tests/aggregate_result_jsons.py experimental-results
   ```

2. This script will generate several outputs:
   - A summary of the experiment results printed to the console
   - Bar charts comparing model performance:
     - One chart for each trial
     - One chart showing the average across all trials
   - An aggregated JSON file containing all results

3. The bar charts will be saved as PNG files:
   - `model_comparison.png` for the average across all trials
   - `model_comparison_trial-X.png` for each individual trial

4. The aggregated results will be saved as `aggregated_results.json` in the `experimental-results` directory.

5. The bar charts provide a visual comparison of different models and providers across various test types. They show:
   - Performance for each test type
   - Comparison between DSPy and f-string implementations
   - Results for different models and providers

6. You can use these visualizations to quickly identify:
   - Which models perform best for each test type
   - How DSPy compares to f-string implementations
   - Any significant differences between trials

Remember to run this script after completing your experiments to get a comprehensive view of your results.