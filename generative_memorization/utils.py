import pandas as pd
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import torch
import numpy as np
import ot  # Optimal Transport library
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# Device configuration: use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load BERT embedding model
bert_model = SentenceTransformer("neuml/pubmedbert-base-embeddings").to(device)

# Load tokenizer and causal model with quantization for memory efficiency
model_id = "ruslanmv/llama3-8B-medical"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
llama_model = AutoModelForCausalLM.from_pretrained(model_id, config=quantization_config)

# Function to measure distance between sequences using Earth Mover's Distance (EMD)
def measure_distance(s_true, s_pred, embedded=False, model='bert', time_weights=None):
    """
    Measures distance between two sequences using Earth Mover's Distance (EMD).

    Args:
        s_true (list or np.array): Ground truth sequence.
        s_pred (list or np.array): Predicted sequence.
        embedded (bool): Whether the sequences are pre-embedded.
        model (str): The model type ('bert' or 'llama').
        time_weights (list): Optional time-based weights for EMD.

    Returns:
        float: Calculated EMD value.
    """
    max_len = max(len(s_true), len(s_pred))
    pairwise_distances = np.ones((max_len, max_len))  # Initialize distance matrix
    time_dist = np.ones((max_len, max_len))  # Initialize time weights matrix

    # Calculate time-based weights if provided
    if time_weights is not None:
        time_dist[:len(s_true), :len(s_pred)] = np.abs(
            np.array(time_weights[0]).reshape(-1, 1) - np.array(time_weights[1]).reshape(1, -1)
        )

    # Calculate pairwise distances
    if embedded:
        dot_product = s_true @ s_pred.T
        norms_A = np.linalg.norm(s_true, axis=1)
        norms_B = np.linalg.norm(s_pred, axis=1)
        norm_matrix = np.outer(norms_A, norms_B)
        cosine_similarity = dot_product / (norm_matrix + 1e-10)
        pairwise_distances[:len(s_true), :len(s_pred)] = 1 - cosine_similarity
    else:
        for i, a_i in enumerate(s_true):
            for j, a_j in enumerate(s_pred):
                if model == 'llama':
                    pairwise_distances[i, j] = 1 - get_embed_sim(
                        text_1=a_i, text_2=a_j, embed_model=None, causal_model=llama_model, tokenizer=tokenizer
                    )
                elif model == 'bert':
                    pairwise_distances[i, j] = 1 - get_embed_sim(
                        text_1=a_i, text_2=a_j, embed_model=bert_model, causal_model=None, tokenizer=None
                    )

    # Uniform weights for EMD calculation
    weights = np.ones(max_len) / max_len
    if time_weights is None:
        emd = ot.emd2(weights, weights, pairwise_distances)
    else:
        emd = ot.emd2(weights, weights, pairwise_distances * (time_dist + 1))
    
    return emd


# Function to get sentence embeddings using BERT or LLaMA
def get_embed(text, model='bert'):
    """
    Generates sentence embeddings using the specified model.

    Args:
        text (str or list): Input text(s).
        model (str): The model type ('bert' or 'llama').

    Returns:
        torch.Tensor: Sentence embeddings.
    """
    if model == 'llama':
        tokenized_input = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            last_hidden_state = llama_model(**tokenized_input, output_hidden_states=True).hidden_states[-1]
        weights_for_non_padding = tokenized_input.attention_mask * torch.arange(
            start=1, end=last_hidden_state.shape[1] + 1
        ).unsqueeze(0)
        sum_embeddings = torch.sum(last_hidden_state * weights_for_non_padding.unsqueeze(-1), dim=1)
        num_non_padding_tokens = torch.sum(weights_for_non_padding, dim=-1).unsqueeze(-1)
        sentence_embeddings = sum_embeddings / num_non_padding_tokens
    elif model == 'bert':
        sentence_embeddings = bert_model.encode(text)
        sentence_embeddings = torch.stack([torch.Tensor(e) for e in sentence_embeddings])
    return sentence_embeddings


# Function to compute cosine similarity between two texts
def get_embed_sim(text_1, text_2, embed_model=None, causal_model=None, tokenizer=None):
    """
    Computes cosine similarity between two text strings.

    Args:
        text_1 (str): First text string.
        text_2 (str): Second text string.
        embed_model: Sentence embedding model.
        causal_model: Causal model for embeddings.
        tokenizer: Tokenizer for causal model.

    Returns:
        float: Cosine similarity score.
    """
    if embed_model is not None:
        embeddings = embed_model.encode([text_1, text_2])
        embeddings_tensor = torch.tensor(embeddings)
        sim = torch.nn.functional.cosine_similarity(embeddings_tensor[0], embeddings_tensor[1], dim=-1).item()
    else:
        assert tokenizer is not None, "Tokenizer is required for causal model."
        tokenized_input = tokenizer([text_1, text_2], padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            last_hidden_state = causal_model(**tokenized_input, output_hidden_states=True).hidden_states[-1]
        weights_for_non_padding = tokenized_input.attention_mask * torch.arange(
            start=1, end=last_hidden_state.shape[1] + 1
        ).unsqueeze(0)
        sum_embeddings = torch.sum(last_hidden_state * weights_for_non_padding.unsqueeze(-1), dim=1)
        num_non_padding_tokens = torch.sum(weights_for_non_padding, dim=-1).unsqueeze(-1)
        sentence_embeddings = sum_embeddings / num_non_padding_tokens
        sim = torch.nn.functional.cosine_similarity(sentence_embeddings[0], sentence_embeddings[1], dim=-1).item()
    return sim


def remove_pad(seq):
    try:
        index = seq.index("[PAD]")
        seq=seq[:index]
    except:
        seq=seq
    return seq


def token_convertor(token_dict, seq):
    code_list = token_dict.keys()
    mapped_list = [token_dict[element]["label"] if (element in code_list and "label" in token_dict[element].keys()) else element for element in seq]
    return mapped_list


def find_codes(conditions):
    f = open("/projects/ehrmamba_memorization/ehrmamba2/meta_vocab.json") 
    data = json.load(f)
    code_dict = dict([(condition,[]) for condition in conditions])
    for code,elem in data.items():
        for condition in conditions:
            if "label" in list(elem.keys()) and condition in elem["label"]:
                code_dict[condition].append(code)
    print(code_dict)
    return code_dict


def generate_time_counter(code_list):
    result = []
    t = 0
    for c_ind, c in enumerate(code_list):
        if c=='TIME//10-20' or c=='TIME//20-40' or c=='TIME//40-60':
            t += 0.1
        elif c=='TIME//0':
            t += 0.2
        elif c=='TIME//1':
            t += 0.4
        elif c=='TIME//2':
            t += 0.9 
        elif 'TIME' in c:
            t += 1.5
        result.append(t) 
    return result
