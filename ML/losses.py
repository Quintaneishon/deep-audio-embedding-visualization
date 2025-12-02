"""
Loss functions for contrastive learning.

Implements Supervised Contrastive Loss (SupCon) for learning embeddings
with genre supervision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss.
    
    Based on: "Supervised Contrastive Learning" (Khosla et al., NeurIPS 2020)
    https://arxiv.org/abs/2004.11362
    
    This loss pulls together embeddings from the same class (positive pairs)
    and pushes apart embeddings from different classes (negative pairs).
    
    The embeddings are expected to be L2 normalized, so that cosine similarity
    equals the dot product.
    """
    
    def __init__(self, temperature=0.07, contrast_mode='all'):
        """
        Initialize Supervised Contrastive Loss.
        
        Args:
            temperature: Temperature parameter for scaling similarities (default: 0.07)
            contrast_mode: 'all' uses all samples as contrast, 'one' uses one positive
        """
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
    
    def forward(self, features, labels):
        """
        Compute supervised contrastive loss.
        
        Args:
            features: [batch_size, embedding_dim] - L2 normalized embeddings
            labels: [batch_size] - integer class labels
        
        Returns:
            loss: Scalar tensor with the contrastive loss
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Ensure features are normalized
        features = F.normalize(features, p=2, dim=1)
        
        # Compute similarity matrix: cosine similarity = dot product (since normalized)
        # [batch_size, batch_size]
        similarity_matrix = torch.matmul(features, features.T)
        
        # Scale by temperature
        similarity_matrix = similarity_matrix / self.temperature
        
        # Create mask for positive pairs (same label)
        # [batch_size, batch_size]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Mask out self-contrast (diagonal)
        logits_mask = torch.ones_like(mask).to(device)
        logits_mask.fill_diagonal_(0)
        
        # Apply mask to remove self-contrast
        mask = mask * logits_mask
        
        # Compute log probabilities
        # Subtract max for numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # Compute log-sum-exp of negative pairs
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
        # Compute mean of log-likelihood over positive pairs
        # Handle case where a class has only one sample in the batch
        mask_sum = mask.sum(1)
        mask_sum = torch.clamp(mask_sum, min=1.0)  # Avoid division by zero
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        
        # Loss is negative log-likelihood
        loss = -mean_log_prob_pos
        loss = loss.mean()
        
        return loss
        