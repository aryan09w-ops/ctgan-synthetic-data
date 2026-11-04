"""
Synthetic Data Generation using CTGAN
Dataset: Wisconsin Breast Cancer (scikit-learn)

What this script does:
1. Loads the breast-cancer dataset.
2. Splits it into real training and test data.
3. Trains a CTGAN on the real training data.
4. Generates synthetic training records.
5. Compares diagnosis proportions and mean-radius distributions.
6. Evaluates Random Forest models trained on real data and on real plus synthetic data.
7. Plots Generator and Discriminator losses.
8. Saves synthetic data and evaluation results as CSV files.

Run:
pip install -r requirements.txt
python synthdata_ai.py
"""

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pandas as pd
from ctgan import CTGAN
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


# 1. Load and prepare the dataset
print("\n[1/7] Loading breast-cancer dataset...")
dataset = load_breast_cancer()
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)

# In sklearn: 0 = malignant, 1 = benign
df = X.copy()
df["diagnosis"] = pd.Series(dataset.target).map({
    0: "malignant",
    1: "benign",
}).values

print("Dataset shape:", df.shape)
print("\nOriginal diagnosis counts:")
print(df["diagnosis"].value_counts())


# 2. Split into train and test
print("\n[2/7] Splitting data...")
train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["diagnosis"],
)

# Keep a separate copy of the target and features for evaluation.
X_train_real = train_df.drop(columns=["diagnosis"])
y_train_real = train_df["diagnosis"]
X_test = test_df.drop(columns=["diagnosis"])
y_test = test_df["diagnosis"]

print("Real training rows:", len(train_df))
print("Testing rows:", len(test_df))
print("\nReal training diagnosis counts:")
print(y_train_real.value_counts())


# 3. Train CTGAN
print("\n[3/7] Training CTGAN...")
print("This may take several minutes depending on your computer.")

# diagnosis is a categorical/discrete column.
ctgan_model = CTGAN(
    epochs=500,
    batch_size=100,
    verbose=True,
)
ctgan_model.fit(
    train_df,
    discrete_columns=["diagnosis"],
)
print("CTGAN training completed.")


# 4. Generate synthetic data
print("\n[4/7] Generating synthetic training data...")
synthetic_train = ctgan_model.sample(len(train_df))

# Make sure the generated diagnosis values are valid.
synthetic_train["diagnosis"] = synthetic_train["diagnosis"].astype(str).str.lower()
synthetic_train = synthetic_train[
    synthetic_train["diagnosis"].isin(["benign", "malignant"])
].copy()

# CTGAN should normally return the requested number of rows.
# If invalid labels were returned, sample extra rows to replace them.
while len(synthetic_train) < len(train_df):
    extra = ctgan_model.sample(len(train_df) - len(synthetic_train))
    extra["diagnosis"] = extra["diagnosis"].astype(str).str.lower()
    extra = extra[extra["diagnosis"].isin(["benign", "malignant"])]
    synthetic_train = pd.concat(
        [synthetic_train, extra],
        ignore_index=True,
    )

synthetic_train = synthetic_train.iloc[:len(train_df)].copy()

print("\nFirst 10 synthetic records:")
print(synthetic_train.head(10).to_string(index=False))
print("\nGenerated diagnosis counts:")
print(synthetic_train["diagnosis"].value_counts())

synthetic_train.to_csv("synthetic_training_data.csv", index=False)
print("\nSaved: synthetic_training_data.csv")


# 5. Evaluate real-only vs. real-plus-synthetic
print("\n[5/7] Evaluating classification models...")
X_train_synthetic = synthetic_train.drop(columns=["diagnosis"])
y_train_synthetic = synthetic_train["diagnosis"]

# Ensure the synthetic feature columns have the same order as real data.
X_train_synthetic = X_train_synthetic[X_train_real.columns]

# Random Forest trained only on real data.
real_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
)
real_model.fit(X_train_real, y_train_real)

# Random Forest trained on real + synthetic data.
X_train_combined = pd.concat(
    [X_train_real, X_train_synthetic],
    ignore_index=True,
)
y_train_combined = pd.concat(
    [y_train_real, y_train_synthetic],
    ignore_index=True,
)

combined_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
)
combined_model.fit(X_train_combined, y_train_combined)

# Both models are evaluated on the same untouched real test set.
real_predictions = real_model.predict(X_test)
combined_predictions = combined_model.predict(X_test)


def calculate_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (malignant)": precision_score(
            y_true,
            y_pred,
            pos_label="malignant",
            zero_division=0,
        ),
        "Recall (malignant)": recall_score(
            y_true,
            y_pred,
            pos_label="malignant",
            zero_division=0,
        ),
        "F1 (malignant)": f1_score(
            y_true,
            y_pred,
            pos_label="malignant",
            zero_division=0,
        ),
    }


results = pd.DataFrame({
    "Real only": calculate_metrics(y_test, real_predictions),
    "Real + synthetic": calculate_metrics(y_test, combined_predictions),
}).T

print("\nEvaluation results:")
print(results.round(3).to_string())

results.to_csv("evaluation_results.csv")
print("\nSaved: evaluation_results.csv")


# 6. Compare real and synthetic distributions
print("\n[6/7] Creating real-vs-synthetic comparison plots...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Diagnosis proportions
diagnosis_plot = pd.DataFrame({
    "Real training data": train_df["diagnosis"].value_counts(normalize=True),
    "Synthetic data": synthetic_train["diagnosis"].value_counts(normalize=True),
}).fillna(0)

diagnosis_plot.plot(
    kind="bar",
    ax=axes[0],
    color=["#2878B5", "#F28E2B"],
)
axes[0].set_title("Diagnosis proportions")
axes[0].set_xlabel("Diagnosis")
axes[0].set_ylabel("Proportion")
axes[0].tick_params(axis="x", rotation=90)
axes[0].legend()

# Mean radius distribution
feature = "mean radius"
axes[1].hist(
    train_df[feature],
    bins=20,
    alpha=0.6,
    density=True,
    label="Real",
    color="#2878B5",
)
axes[1].hist(
    synthetic_train[feature],
    bins=20,
    alpha=0.6,
    density=True,
    label="Synthetic",
    color="#F28E2B",
)
axes[1].set_title("Mean radius distribution")
axes[1].set_xlabel("Mean radius")
axes[1].set_ylabel("Density")
axes[1].legend()

plt.tight_layout()
plt.savefig("real_vs_synthetic_comparison.png", dpi=300)
plt.show()


# 7. Plot CTGAN generator/discriminator losses
print("\n[7/7] Plotting CTGAN training losses...")
try:
    loss_df = ctgan_model.loss_values

    if loss_df is None or len(loss_df) == 0:
        print("No CTGAN loss history was exposed by this CTGAN version.")
    else:
        print("\nLast CTGAN loss records:")
        print(loss_df.tail().to_string(index=False))

        # CTGAN versions commonly expose columns:
        # Epoch, Generator Loss, Discriminator Loss
        generator_col = next(
            (
                column
                for column in loss_df.columns
                if "generator" in column.lower()
            ),
            None,
        )
        discriminator_col = next(
            (
                column
                for column in loss_df.columns
                if "discriminator" in column.lower()
            ),
            None,
        )

        if generator_col and discriminator_col:
            plt.figure(figsize=(12, 5))
            plt.plot(
                loss_df[generator_col].values,
                label="Generator Loss",
            )
            plt.plot(
                loss_df[discriminator_col].values,
                label="Discriminator Loss",
            )
            plt.title("Generator and Discriminator Loss During CTGAN Training")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(alpha=0.2)
            plt.tight_layout()
            plt.savefig("ctgan_training_losses.png", dpi=300)
            plt.show()
        else:
            print("Could not identify generator/discriminator loss columns.")
            print("Available columns:", list(loss_df.columns))

except Exception as error:
    print("Could not plot CTGAN losses:", error)


print("\nAll steps completed.")
print("Generated files:")
print("- synthetic_training_data.csv")
print("- evaluation_results.csv")
print("- real_vs_synthetic_comparison.png")
print("- ctgan_training_losses.png (if loss history is available)")
