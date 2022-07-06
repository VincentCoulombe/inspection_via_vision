import torch

def iou(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    intersection = torch.logical_and(labels, outputs)
    union = torch.logical_or(labels, outputs)
    return float(torch.sum(intersection) / torch.sum(union))


if __name__ == '__main__':
    pass