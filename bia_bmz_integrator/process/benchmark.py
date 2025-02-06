import numpy as np
from monai.metrics import DiceMetric, MeanIoU, PSNRMetric, RMSEMetric, SSIMMetric
import torch


# Functions to benchmark results
def segmentation_scores(pred_mask, groundtruth_mask, t_binary_output, t_reference_binary):
    true_positives = np.sum(pred_mask*groundtruth_mask)
    true_false_postives = np.sum(pred_mask)
    precision = np.mean(true_positives/true_false_postives)
    #print(f"Precision score: {precision}")

    total_pixel_truth = np.sum(groundtruth_mask)
    recall = np.mean(true_positives/total_pixel_truth)
    #print(f"Recall score: {recall}")

    iou_metric = MeanIoU(include_background=True, reduction="mean")
    iou_score = iou_metric(y_pred=t_binary_output, y=t_reference_binary)
    iou_score = torch.IntTensor.item(iou_score)
    #print(f"IoU score: {iou_score}")

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_score = dice_metric(y_pred=t_binary_output, y=t_reference_binary)
    dice_score = torch.IntTensor.item(dice_score)
    #print(f"Dice score: {dice_score}")
    
    scores = [round(precision, 3),round(recall, 3),round(iou_score, 3),round(dice_score, 3)]
    
    return scores

def restoration_scores(t_output, t_reference, max_val):

    psnr_metric = PSNRMetric(max_val, reduction="mean", get_not_nans=False)
    psnr_score = psnr_metric(y_pred=t_output, y=t_reference)
    psnr_score = torch.IntTensor.item(psnr_score)
    #print(f"PSNR score reconstructed image: {psnr_score}")
    
    rmse_metric = RMSEMetric(reduction="mean", get_not_nans=False)
    rmse_score = rmse_metric(y_pred=t_output, y=t_reference)
    rmse_score = torch.IntTensor.item(rmse_score)
    nrmse_score = rmse_score/max_val
    #print(f"NRMSE score reconstructed image: {nrmse_score}")

    ssim_metric = SSIMMetric(spatial_dims=2, data_range=max_val)
    ssim_score = ssim_metric(y_pred=t_output[0, :, :, :, :], y=t_reference[0, :, :, :, :])
    ssim_score = torch.IntTensor.item(ssim_score)
    #print(f"SSIM score reconstructed image: {ssim_score}")

    scores = [round(psnr_score,3),round(nrmse_score,3),round(ssim_score,3)]

    return scores

