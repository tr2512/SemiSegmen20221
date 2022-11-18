import torch

def IoU(output, target, c, ignore_index=255):
    assert output.shape == target.shape
    mask = (target != ignore_index)
    output = output[mask]
    target = target[mask]
    intersect = output[output == target]
    area_intersect = torch.histc(
        intersect.float(), bins=(c), min=0, max=c - 1)
    area_pred_label = torch.histc(
        output.float(), bins=(c), min=0, max=c - 1)
    area_label = torch.histc(
        target.float(), bins=(c), min=0, max=c - 1)
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union

def OverallAcc(output, target, ignore_index=255):
    assert output.shape == target.shape
    mask = (target != ignore_index)
    output = output[mask]
    target = target[mask]
    return (output == target).sum().item(), torch.numel(output)