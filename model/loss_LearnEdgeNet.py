import torch.nn.functional as F

def bce_with_logits(output, target, pos_weight=None):
    # Ensure target is on the same device as output
    target = target.to(output.device)

    if pos_weight is not None:
        # Ensure pos_weight is also moved to GPU
        pos_weight = pos_weight.to(output.device)

    return F.binary_cross_entropy_with_logits(output, target, pos_weight=pos_weight)
