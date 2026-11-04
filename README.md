# CTGAN Synthetic Data Generation

This project uses a Conditional Tabular GAN (CTGAN) to generate synthetic records from the Wisconsin Breast Cancer dataset included in scikit-learn. It compares a Random Forest trained on real data with one trained on real and synthetic data.

## What the script does

- Loads and splits the dataset into training and test sets.
- Trains CTGAN on the training data and generates synthetic records.
- Compares diagnosis proportions and mean-radius distributions.
- Evaluates Random Forest models on the untouched real test set.
- Saves synthetic data, evaluation results, and plots.

## Requirements

Install the required Python packages:

```bash
pip install pandas matplotlib scikit-learn ctgan
