import numpy as np
import pandas as pd
import torch
import polars
from utils import meausre_distance, get_embed, generate_time_counter, remove_pad, token_convertor

# Set device to GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def trajectory_memorization_test(
    ground_truth_path,
    generated_data_path,
    exp_setup,
    token_freq=None,
    token_dict=None,
    split=None,
    n_prompts=None,
    trajectory_lens=None,
    subgroup_ids=None
):
    """
    Test for trajectory memorization by comparing ground truth labels with generated predictions.
    
    Args:
        ground_truth_path (str): Path to the ground truth dataset (Parquet file).
        generated_data_path (str): Path to the generated data (Parquet file). Optional for "random" setup.
        exp_setup (str): Experiment setup type ("random", "static_prompt", "N_prompt").
        token_freq (dict): Token frequency dictionary for random baseline generation (used in "random" setup).
        token_dict (dict): Dictionary for converting token codes to human-redable labels.
        split (str): Dataset split to use (e.g., "train", "test").
        n_prompts (int): Number of prompts to skip in "N_prompt" setup.
        trajectory_lens (list): List of trajectory lengths for comparison.
        subgroup_ids (list): Subgroup of patient IDs to include in the test.
    
    Returns:
        dict: Dictionary of trajectory lengths mapped to mean and standard deviation of distances.
    """
    # Load ground truth data
    df_gt = polars.read_parquet(ground_truth_path).to_pandas()
    df_gt = df_gt.drop_duplicates(subset=["patient_id"])
    patient_ids = np.unique(df_gt["patient_id"].to_list())
    distance = {t_len: [] for t_len in trajectory_lens}

    # Random setup: Generate or use random predictions
    if exp_setup == "random":
        if generated_data_path is None:
            # Generate most likely baseline from token frequencies
            top_code_values = np.array(list(token_freq.values())[:5000])
            top_codes = list(token_freq.keys())[:5000]
            categorical_dist = torch.distributions.Categorical(
                torch.Tensor(top_code_values / np.sum(top_code_values))
            )
            samples = [
                np.array([top_codes[d] for d in categorical_dist.sample((500,))])
                for _ in range(1000)
            ]
            time_weight_predicted = np.zeros((800, 500))
        else:
            # Use randomly generated samples
            df = polars.read_parquet(generated_data_path).to_pandas()
            samples = [df["predicted_tokens"].iloc[sid].tolist() for sid in range(len(df))]
            time_weight_predicted = [
                generate_time_counter(p_labels) for p_labels in samples
            ]

        # Get embeddings for generated samples
        samples_embedding = [get_embed(sample, model="bert") for sample in samples]
        if split is not None:
            df_gt = df_gt[df_gt["split"] == split]

        # Compare with ground truth for each patient
        for pid in patient_ids:
            distance_p = {t_len: [] for t_len in trajectory_lens}
            df_p = df_gt[df_gt["patient_id"] == pid]
            actual_labels = remove_pad(df_p["event_tokens"].iloc[0])

            if trajectory_lens is not None:
                trajectory_len = np.max(trajectory_lens)
                actual_labels = actual_labels[:min(trajectory_len, len(actual_labels))]

            time_weight_gt = generate_time_counter(actual_labels)
            actual_labels_embeddings = get_embed(actual_labels, model="bert")

            # Calculate distance for each trajectory length
            for sid, sample in enumerate(samples_embedding):
                for t_lens in trajectory_lens:
                    dist = meausre_distance(
                        s_true=actual_labels_embeddings[:t_lens],
                        s_pred=sample[:t_lens],
                        embedded=True,
                        time_weights=[
                            time_weight_gt[:t_lens],
                            time_weight_predicted[sid][:t_lens],
                        ],
                    )
                    distance_p[t_lens].append(dist)

            for t_lens in trajectory_lens:
                distance[t_lens].extend(
                    [(np.mean(distance_p[t_lens]), np.std(distance_p[t_lens]))]
                )

    # Static or N-prompt setup
    elif exp_setup in ["static_prompt", "N_prompt"]:
        df = polars.read_parquet(generated_data_path).to_pandas()
        df = df[df["split"] == split]

        if subgroup_ids is not None:
            df = df[df["patient_id"].isin(subgroup_ids)]

        pid_generated = np.unique(df["patient_id"].to_list())
        for pid in pid_generated:
            if pid not in patient_ids:
                continue

            df_p = df_gt[df_gt["patient_id"] == pid]
            actual_labels = remove_pad(df_p["event_tokens"].iloc[0])
            actual_labels = actual_labels[(n_prompts if exp_setup == "N_prompt" else 2):]

            if trajectory_lens is not None:
                trajectory_len = np.max(trajectory_lens)
                actual_labels = actual_labels[:min(trajectory_len, len(actual_labels))]

            actual_labels = token_convertor(token_dict, actual_labels)
            time_weight_gt = generate_time_counter(actual_labels)
            df_t = df[df["patient_id"] == pid]
            trajectories = df_t["trajectory"].to_list()
            actual_labels_embeddings = get_embed(actual_labels, model="bert")

            distance_p = {t_len: [] for t_len in trajectory_lens}
            for sid in trajectories:
                predicted_labels = remove_pad(df_t["predicted_tokens"].iloc[sid])

                if trajectory_lens is not None:
                    trajectory_len = np.max(trajectory_lens)
                    predicted_labels = predicted_labels[:min(trajectory_len, len(predicted_labels))]

                predicted_labels = token_convertor(token_dict, predicted_labels)
                time_weight_predicted = generate_time_counter(predicted_labels)
                predicted_labels_embeddings = get_embed(predicted_labels, model="bert")

                for t_lens in trajectory_lens:
                    dist = meausre_distance(
                        s_true=actual_labels_embeddings[:t_lens],
                        s_pred=predicted_labels_embeddings[:t_lens],
                        embedded=True,
                        time_weights=[
                            time_weight_gt[:t_lens],
                            time_weight_predicted[:t_lens],
                        ],
                    )
                    distance_p[t_lens].append(dist)

            for t_lens in trajectory_lens:
                distance[t_lens].extend(
                    [(np.mean(distance_p[t_lens]), np.std(distance_p[t_lens]))]
                )

    return distance
