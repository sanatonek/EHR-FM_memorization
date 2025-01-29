import numpy as np
import pandas as pd
import polars
import pickle
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, precision_score, recall_score,
    roc_auc_score, average_precision_score
)

def probing_test(file_names, nprompts, mapping_df, output_path, class_weights=None):
    """
    Runs a probing test using an MLP classifier for each dataset.

    Parameters:
    - file_names (dict): Dictionary where keys are condition names and values are dataset file paths (Parquet format).
    - nprompts (int): Number of prompts used in data collection.
    - mapping_df (pd.DataFrame): DataFrame containing patient_id mapping to splits (e.g., 'pretrain', 'train', 'test').
    - output_path (str): Directory where results and trained models are saved.
    - class_weights (dict, optional): Dictionary for class weights to handle imbalanced data.

    Flexibilities:
    - Allows training on different fractions (e.g., 10%, 20%, 50%) of the dataset.
    - Supports weighting of training data to mitigate class imbalance.
    - Filters data to use only pretrain split, ensuring correct separation of training and test sets.
    - Enables hyperparameter tuning using GridSearchCV for optimal model selection.
    - Saves trained models and results for later analysis.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    results = []
    
    for sensitive_attribute, file_path in file_names.items():
        print(f"Processing: {sensitive_attribute} --- {nprompts} Prompts")

        # Load and merge data
        df = polars.read_parquet(file_path).to_pandas()[['patient_id', 'event_token_embeddings']]
        df = df.merge(mapping_df, on='patient_id', how='inner')
        print('Size of merged df:', len(df))

        # Filter pretrain data
        df_pretrain = df[df['split'] == 'pretrain']
        if df_pretrain.empty:
            print(f"Skipping {sensitive_attribute}, no 'pretrain' data available.")
            continue

        X_full = np.stack(df_pretrain['event_token_embeddings'].to_numpy())
        y_full = df_pretrain[sensitive_attribute]

        # Train on different fractions of the pretrain data
        for fraction in [0.1, 0.2, 0.5]:
            print(f"Training with {int(fraction * 100)}% of the pretrain data")

            # Split data into training and test sets while maintaining class balance
            X_train, X_test, y_train, y_test = train_test_split(
                X_full, y_full, test_size=(1 - fraction), random_state=42, stratify=y_full
            )

            print('Training size:', len(y_train), 'Testing size:', len(y_test))

            sample_weights = None
            if class_weights:
                # Apply class weighting for handling imbalanced datasets
                sample_weights = np.array([class_weights.get(label, 1) for label in y_train])

            # Define hyperparameter grid for tuning MLP classifier
            param_grid = {
                "hidden_layer_sizes": [(100,), (100, 100), (100, 100, 100)],
                "alpha": [0.0001, 0.001, 0.01],
                "learning_rate_init": [0.001, 0.01, 0.1],
                "max_iter": [200, 300, 400]
            }

            # Perform hyperparameter tuning using GridSearchCV
            grid_search = GridSearchCV(
                MLPClassifier(random_state=42),
                param_grid=param_grid,
                cv=3,
                scoring="f1_weighted",
                verbose=1
            )
            grid_search.fit(X_train, y_train, sample_weight=sample_weights if sample_weights is not None else None)
            best_params = grid_search.best_params_
            print("Best hyperparameters:", best_params)

            # Train final model with the best hyperparameters
            model = MLPClassifier(
                **best_params, random_state=42
            )
            model.fit(X_train, y_train)

            # Evaluate the model on the test set and store results
            y_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_pred)
            balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            auroc = roc_auc_score(y_test, y_pred)
            auprc = average_precision_score(y_test, y_pred)

            results.append({
                "Condition": sensitive_attribute,
                "Fraction": fraction,
                "Accuracy": balanced_accuracy,
                "AUROC": auroc,
                "AUPRC": auprc,
                "F1": f1,
                "Precision": precision,
                "Recall": recall,
                "Best Params": best_params
            })

            print(f"Results for {sensitive_attribute} with {int(fraction * 100)}% pretrain data:", results[-1])

            # Save the trained model
            joblib.dump(model, f"{output_path}best_model_{sensitive_attribute}_{nprompts}_{int(fraction * 100)}.pkl")

    # Save results to CSV for further analysis
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{output_path}probing_results.csv", index=False)

    # Print the final results summary
    print(results_df)
