import torch
import torch.nn.functional as F

def label_smoothing_loss(logits, targets, smoothing=0.1):
    """
    logits: (batch_size, num_classes)
    targets: (batch_size,)
    """
    num_classes = logits.size(1)

    with torch.no_grad():
        true_dist = torch.zeros_like(logits)
        true_dist.fill_(smoothing / (num_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)

    log_probs = F.log_softmax(logits, dim=1)
    loss = -(true_dist * log_probs).sum(dim=1).mean()
    return loss


if __name__ == "__main__":
    logits = torch.randn(4, 3)
    targets = torch.tensor([0, 1, 2, 1])

    loss = label_smoothing_loss(logits, targets)
    print(loss)


"""
📌 Label Smoothing:

    Reduce overconfidence

    Improve generalization
"""