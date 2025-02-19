import torch
import numpy as np
from tqdm import tqdm
from typing import List, Tuple
from sklearn.metrics import roc_auc_score
import pandas as pd


class MembershipDetector:
    def __init__(self, k_percent: float = 20.0):
        self.k_percent = k_percent

    def get_sequence_probs(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        device: torch.device
    ) -> List[float]:
        """
        Get token probabilities for a sequence using your model
        """
        token_probs = []
        
        with torch.no_grad():
            # For each position, get probability of next token
            for i in range(input_ids.size(1) - 1):
                # Get input up to current position
                input_slice = input_ids[:, :i+1]
                
                # Get model output for this slice
                outputs = model(input_slice)
                logits = outputs.logits
                
                # Get probabilities for next token
                next_token_probs = torch.softmax(logits[:, -1, :], dim=-1)
                
                # Get probability of actual next token
                actual_next_token = input_ids[:, i+1]
                prob = next_token_probs.gather(1, actual_next_token.unsqueeze(-1))
                token_probs.append(prob.item())
        
        return token_probs

    def compute_sequence_score(self, token_probs: List[float]) -> float:
        """
        Compute MIN-K% PROB score for a sequence
        """
        if not token_probs:
            return 0.0
            
        token_probs = np.array(token_probs)
        log_probs = np.log(token_probs + 1e-10)
        
        # Calculate number of tokens to select (k%)
        k = max(1, int(len(token_probs) * self.k_percent / 100))
        
        # Get indices of k tokens with lowest probabilities
        lowest_k_indices = np.argpartition(token_probs, k)[:k]
        
        return float(np.mean(log_probs[lowest_k_indices]))

    def process_dataloader(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        is_member: bool
    ) -> List[dict]:
        """
        Process a single dataloader and return results
        """
        results = []
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Processing {'member' if is_member else 'non-member'} data")):
            input_ids = batch["concept_ids"].to(device)
            
            # Process each sequence in batch
            for seq_idx in range(input_ids.size(0)):
                # Get probabilities for this sequence
                seq_probs = self.get_sequence_probs(
                    model,
                    input_ids[seq_idx:seq_idx+1],
                    device
                )
                
                # Compute detection score
                score = self.compute_sequence_score(seq_probs)
                
                results.append({
                    'batch_idx': batch_idx,
                    'seq_idx': seq_idx,
                    'is_member': is_member,
                    'detection_score': score,
                    'min_prob': min(seq_probs) if seq_probs else 0,
                    'mean_prob': np.mean(seq_probs) if seq_probs else 0
                })
        
        return results

    def detect_membership(
        self,
        model: torch.nn.Module,
        member_dataloader: torch.utils.data.DataLoader,
        nonmember_dataloader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Tuple[float, pd.DataFrame]:
        """
        Run membership detection using separate member and non-member dataloaders
        """
        model.eval()
        
        with torch.no_grad():
            # Process member data
            member_results = self.process_dataloader(model, member_dataloader, device, True)
            
            # Process non-member data
            nonmember_results = self.process_dataloader(model, nonmember_dataloader, device, False)
        
        # Combine results
        all_results = pd.DataFrame(member_results + nonmember_results)
        
        # Calculate AUC
        y_true = all_results['is_member'].values
        y_score = all_results['detection_score'].values
        auc = roc_auc_score(y_true, y_score)
        
        return auc, all_results


def run_membership_detection(
    model,
    member_dataloader,
    nonmember_dataloader,
    device,
    k_percent=20.0
):
    """
    Run membership detection experiment
    """
    detector = MembershipDetector(k_percent=k_percent)
    
    auc, results = detector.detect_membership(
        model=model,
        member_dataloader=member_dataloader,
        nonmember_dataloader=nonmember_dataloader,
        device=device
    )
    
    print(f"\nDetection Results:")
    print(f"AUC Score: {auc:.3f}")
    
    # Print statistics
    member_scores = results[results['is_member']]['detection_score']
    non_member_scores = results[~results['is_member']]['detection_score']
    
    print("\nMember sequences:")
    print(f"Mean score: {member_scores.mean():.3f}")
    print(f"Min score: {member_scores.min():.3f}")
    print(f"Max score: {member_scores.max():.3f}")
    
    print("\nNon-member sequences:")
    print(f"Mean score: {non_member_scores.mean():.3f}")
    print(f"Min score: {non_member_scores.min():.3f}")
    print(f"Max score: {non_member_scores.max():.3f}")
    
    return auc, results
