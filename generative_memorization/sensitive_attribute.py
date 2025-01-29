import numpy as np
import torch
import polars
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score
from utils import *  # Import custom utility functions

# Set the device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sensitive_attribute_test(
    ground_truth_path,
    generated_data_path,
    exp_setup,
    sensitive_attribute,
    split,
    p_th=0.5,
    subgroup_ids=None
):
    """
    Perform sensitive attribute testing by comparing the presence of sensitive attributes
    in generated predictions with ground truth data.

    Args:
        ground_truth_path (str): Path to the ground truth dataset (Parquet file).
        generated_data_path (str): Path to the generated predictions (Parquet file).
        exp_setup (str): Experiment setup type ("random", "static_prompt", or "N_prompt").
        sensitive_attribute (list): List of sensitive attribute tokens to check.
        split (str): Dataset split to use (e.g., "pretrain", "test").
        p_th (float): Threshold for prediction classification.
        subgroup_ids (list): Optional list of patient IDs for subgroup analysis.

    Returns:
        tuple: AUROC, AUPRC, precision, recall, and true positive predictions with probabilities.
    """
    # Load ground truth data
    df_gt = polars.read_parquet(ground_truth_path).to_pandas()
    df_gt = df_gt.drop_duplicates(subset=["patient_id"])
    patient_ids = np.unique(df_gt["patient_id"].to_list())
    obs_lens, total_lens = [], []

    if exp_setup == "random":
        # Random setup: Compare predicted tokens against sensitive attributes
        df = polars.read_parquet(generated_data_path).to_pandas()
        tp_pid = []
        df_gt = df_gt[df_gt["split"] == split]
        patient_ids = np.unique(df_gt["patient_id"].to_list())

        pred_count = 0
        labels = []

        # Iterate through generated samples
        for sid in range(len(df)):
            predicted_labels = df["predicted_tokens"].iloc[sid].tolist()
            try:
                # Remove padding tokens
                index = predicted_labels.index("[PAD]")
                predicted_labels = predicted_labels[:index]
                obs_lens.append(len(predicted_labels[:index]))
            except ValueError:
                obs_lens.append(len(predicted_labels))
            
            total_lens.append(len(predicted_labels))

            # Count presence of sensitive attributes
            if any(att in predicted_labels for att in sensitive_attribute):
                pred_count += 1

        # Collect ground truth labels
        for pid in patient_ids:
            df_p = df_gt[df_gt["patient_id"] == pid]
            actual_labels = df_p["event_tokens"].iloc[0].tolist()
            labels.append(int(any(att in actual_labels for att in sensitive_attribute)))

        # Calculate predictions
        predictions = [pred_count / len(df)] * len(labels)

    elif exp_setup in ["static_prompt", "N_prompt"]:
        # Static or N-prompt setup
        df = polars.read_parquet(generated_data_path).to_pandas()
        df = df[df["split"] == split]
        if subgroup_ids is not None:
            df = df[df["patient_id"].isin(subgroup_ids)]
        labels, predictions = [], []
        tp_pid = []
        pid_generated = np.unique(df["patient_id"].to_list())

        # Iterate through generated patient data
        for pid in pid_generated:
            if pid not in patient_ids:
                print("Patient ID not found in ground truth:", pid)
                continue

            df_p = df_gt[df_gt["patient_id"] == pid]
            actual_labels = remove_pad(df_p["event_tokens"].iloc[0])
            df_t = df[df["patient_id"] == pid]
            trajectories = df_t["trajectory"].to_list()

            pred_count = 0

            # Iterate through predicted trajectories
            for sid in trajectories:
                predicted_labels = remove_pad(df_t["predicted_tokens"].iloc[sid])
                if len(predicted_labels) < 10:
                    continue
                total_lens.append(len(predicted_labels))
                if any(att in predicted_labels for att in sensitive_attribute):
                    pred_count += 1

            labels.append(int(any(att in actual_labels for att in sensitive_attribute)))
            predictions.append(pred_count / len(trajectories))

            # Collect true positive predictions above the threshold
            if labels[-1] == 1 and predictions[-1] > p_th:
                tp_pid.append((pid, predictions[-1]))

    # Print results
    print("Number of samples:", len(labels))
    print("Positive ratio:", np.sum(labels) / len(labels))
    print("True positive predictions (PIDs):", tp_pid)

    # Calculate evaluation metrics
    auroc = roc_auc_score(np.array(labels), np.array(predictions))
    auprc = average_precision_score(np.array(labels), np.array(predictions))
    precision = precision_score(np.array(labels), (np.array(predictions) > p_th).astype(int))
    recall = recall_score(np.array(labels), (np.array(predictions) > p_th).astype(int))

    print("Total samples:", len(labels))
    return auroc, auprc, precision, recall, tp_pid
