import numpy as np
import matplotlib.pyplot as plt
import zarr
import dask.array as da
import bioimageio.core
from bioimageio.core import Tensor, Sample, test_model, create_prediction_pipeline
from bioimageio.spec.utils import load_array
from bioimageio.spec.model import v0_5, v0_4
from monai.metrics import DiceMetric, MeanIoU, PSNRMetric, RMSEMetric, SSIMMetric, ConfusionMatrixMetric, get_confusion_matrix, compute_confusion_matrix_metric
import torch
import pandas as pd
import warnings
import cv2

# Function to create appropiate model inputs from zarr

def remote_zarr_to_model_input(ome_zarr_uri):
    """Makes many bad assumptions about shape."""
    
    zgroup = zarr.open(ome_zarr_uri)
    zarray = zgroup['0']    

    darray = da.from_zarr(zarray)

    return darray

# Function to crop images 
def crop_center(array,window):
    dims = array.shape
    x = dims[-1]
    y = dims[-2] 
    if (x>window[0]) & (y>window[1]):
        startx = x//2-(window[0]//2)
        starty = y//2-(window[1]//2)  
        return array[:,:,:,starty:starty+window[1],startx:startx+window[0]]
    else:
        print("reshape window values must be smaller than image dimensions")
 
# Function to reorder tensor dimensions to match OME.Zarr dimension order: "TCZYX". Returns an array
def reorder_array_dimensions(in_tensor, in_id):
    in_array = np.asarray(in_tensor.members[in_id].data)
    in_dim = in_tensor.members[in_id].dims
    if 'channel' in in_dim:
        bch_index = in_dim.index('channel')
    elif 'c' in in_dim:
        bch_index = in_dim.index('c')
    x_index = in_dim.index('x')
    y_index = in_dim.index('y') 

    if 'z' in in_dim:
        z_index = in_dim.index('z') 
        array_tczxy = np.transpose(in_array, (0, bch_index, z_index, y_index, x_index))
    else:
        array_tczxy = np.transpose(in_array, (0, bch_index, y_index, x_index))
        array_tczxy = np.expand_dims(array_tczxy, 2)

    return array_tczxy 

# Function to handle model inference
def run_model_inference(bmz_model, arr):
    # load model
    model_resource = bioimageio.core.load_description(bmz_model)

    if isinstance(model_resource, v0_5.ModelDescr):
        test_input_image = load_array(model_resource.inputs[0].test_tensor)
        # match test data type with the data type of the model input
        arr = arr.astype(test_input_image.dtype)

        # Reshape data to match model input dimensions
        # new_np_array = np.squeeze(arr).compute()
        indices = [i for i in range(len(test_input_image.shape)) if test_input_image.shape[i] == 1]
        right_dims = np.expand_dims(arr, indices)
        input_tensor = Tensor.from_numpy(right_dims, dims=tuple(model_resource.inputs[0].axes))

        # Create collection of tensors (sample)
        inp_id = model_resource.inputs[0].id
        outp_id = model_resource.outputs[0].id
        sample = Sample(members={inp_id: input_tensor}, stat={}, id="id")


    elif isinstance(model_resource, v0_4.ModelDescr):
        test_input_image = load_array(model_resource.test_inputs[0])

        arr = arr.astype(test_input_image.dtype)
        #arr = arr.astype('float')

        #new_np_array = np.squeeze(arr).compute()
        indices = [i for i in range(len(test_input_image.shape)) if test_input_image.shape[i] == 1]
        right_dims = np.expand_dims(arr, indices)
        input_tensor = Tensor.from_numpy(right_dims, dims=tuple(model_resource.inputs[0].axes))

        # Create collection of tensors (sample)
        inp_id = model_resource.inputs[0].name
        outp_id = model_resource.outputs[0].name
        sample = Sample(members={inp_id: input_tensor}, stat={}, id="id")
    
    else:
        raise ValueError("This model specification version is not supported")

    prediction_pipeline = create_prediction_pipeline(model_resource)
    prediction = prediction_pipeline.predict_sample_without_blocking(sample) # works (does not perform tiling)
    #prediction = prediction_pipeline.predict_sample_with_blocking(sample) # doesn't always work. i.e. hiding-blowfish

    return prediction, sample, inp_id, outp_id


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

# Functions to plot the results
def show_images_gt(dask_array, output_array, ref_array, binary_output, ref_array_binary, benchmark_channel=0):
    correct_ch_output = np.take(output_array, [benchmark_channel], 1)
    
    plt.figure()
    ax1 = plt.subplot(2, 3, 1)
    ax1.set_title("Input")
    ax1.axis("off")
    plt.imshow(dask_array[0, 0, 0, :, :])
    
    ax2 = plt.subplot(2, 3, 2)
    ax2.set_title("Prediction")
    ax2.axis("off")
    plt.imshow(correct_ch_output[0, 0, 0, :, :])
    
    ax3 = plt.subplot(2, 3, 3)
    ax3.set_title("Reference annotation")
    ax3.axis("off")
    plt.imshow(ref_array[0, 0, 0, :, :])
    
    ax4 = plt.subplot(2, 3, 4)
    ax4.set_title("Reference binary")
    ax4.axis("off")
    plt.imshow(ref_array_binary[0, 0, 0, :, :])
    
    ax5 = plt.subplot(2, 3, 5)
    ax5.set_title("Prediction binary")
    ax5.axis("off")
    plt.imshow(binary_output[0, 0, 0, :, :])
    plt.show()

def show_images(dask_array, output_array, binary_output, benchmark_channel=0):
    correct_ch_output = np.take(output_array, [benchmark_channel], 1)
    plt.figure()
    ax1 = plt.subplot(1, 3, 1)
    ax1.set_title("Input")
    ax1.axis("off")
    plt.imshow(dask_array[0, 0, 0, :, :])
    
    ax2 = plt.subplot(1, 3, 2)
    ax2.set_title("Prediction")
    ax2.axis("off")
    plt.imshow(correct_ch_output[0, 0, 0, :, :])
     
    ax5 = plt.subplot(1, 3, 3)
    ax5.set_title("Prediction binary")
    ax5.axis("off")
    plt.imshow(binary_output[0, 0, 0, :, :])
    plt.show()

def process_run(
    bmz_model, 
    ome_zarr_uri, 
    plot_images=True, 
    crop_image=None, 
    z_slices=None, 
    channel=None, 
    t_slices=None, 
    benchmark_channel=0
) -> dict:
    
    # Load image and annotations
    dask_array = remote_zarr_to_model_input(ome_zarr_uri)

    # Optional cropping
    if crop_image:
        dask_array = crop_center(dask_array, crop_image)

    # Slices and channel selection
    if z_slices:
        dask_array = dask_array[:, :, z_slices[0]:z_slices[1], :, :]

    if channel is not None:
        dask_array = dask_array[:, channel, :, :, :]

    if t_slices:
        dask_array = dask_array[t_slices[0]:t_slices[1], :, :, :, :]

    # Run model inference
    new_np_array = np.squeeze(dask_array).compute()
    prediction, sample, inp_id, outp_id = run_model_inference(bmz_model, new_np_array)

    # Process results for benchmarking
    output_array = reorder_array_dimensions(prediction, outp_id)
    correct_ch_output = np.take(output_array, [benchmark_channel], 1)
    input_array = reorder_array_dimensions(sample, inp_id)
    binary_output = correct_ch_output >= 0.5

    # Plot images
    if plot_images:
        #show_images(dask_array, output_array, ref_array, binary_output, t_reference_binary, benchmark_channel)
        show_images(input_array, output_array, binary_output, benchmark_channel)
    
    output = {}
    output["prediction"] = correct_ch_output

    return output
    

def process_benchmark(
    bmz_model, 
    ome_zarr_uri, 
    reference_annotations, 
    plot_images=True, 
    crop_image=None, 
    z_slices=None, 
    channel=None, 
    t_slices=None, 
    benchmark_channel=0
) -> dict:
    
    # Load image and annotations
    dask_array = remote_zarr_to_model_input(ome_zarr_uri)
    ref_array = remote_zarr_to_model_input(reference_annotations)

    # Optional cropping
    if crop_image:
        dask_array = crop_center(dask_array, crop_image)
        ref_array = crop_center(ref_array, crop_image)

    # Slices and channel selection
    if z_slices:
        dask_array = dask_array[:, :, z_slices[0]:z_slices[1], :, :]
        ref_array = ref_array[:, :, z_slices[0]:z_slices[1], :, :]

    if channel is not None:
        dask_array = dask_array[:, channel, :, :, :]
        ref_array = ref_array[:, channel, :, :, :]
    if t_slices:
        dask_array = dask_array[t_slices[0]:t_slices[1], :, :, :, :]
        ref_array = ref_array[t_slices[0]:t_slices[1], :, :, :, :]

    # Run model inference
    new_np_array = np.squeeze(dask_array).compute()
    prediction, sample, inp_id, outp_id = run_model_inference(bmz_model, new_np_array)

    # Process results for benchmarking
    output_array = reorder_array_dimensions(prediction, outp_id)
    correct_ch_output = np.take(output_array, [benchmark_channel], 1)
    input_array = reorder_array_dimensions(sample, inp_id)
    ref_array = np.asarray(ref_array)
    ref_array = ref_array.astype(output_array.dtype)
    
    if correct_ch_output.shape!=ref_array.shape:
        o_array = np.squeeze(correct_ch_output)
        res_o_array = cv2.resize(o_array,dsize=(ref_array.shape[-2],ref_array.shape[-1]))
        correct_ch_output = np.expand_dims(res_o_array, tuple(np.arange(0, ref_array.ndim-2)))
        warnings.warn(f"reference and prediction shapes dont match, received reference: [{ref_array.shape}] and prediction: [{output_array.shape}]. Prediction has been reshaped")

    # Binarize output and reference arrays
    t_output = torch.from_numpy(correct_ch_output)
    binary_output = correct_ch_output >= 0.5
    t_binary_output = torch.from_numpy(binary_output)
    t_reference = torch.from_numpy(ref_array)
    t_reference_binary = torch.from_numpy(ref_array.astype(bool))
    max_val = np.max(ref_array) - np.min(ref_array)

    # Run benchmarks
    seg_scores = segmentation_scores(binary_output, ref_array.astype(bool), t_binary_output, t_reference_binary)  
    res_scores = restoration_scores(t_output, t_reference, max_val)
    scores = seg_scores + res_scores 

    # Plot images
    if plot_images:
        #show_images(dask_array, output_array, ref_array, binary_output, t_reference_binary, benchmark_channel)
        show_images_gt(input_array, output_array, ref_array, binary_output, t_reference_binary, benchmark_channel)
    
    output = {}
    output["scores"] = scores
    output["prediction"] = correct_ch_output

    return output

def bulk_process(bmz_models, datasets, z_planes=None, crop_image=None, plot_images=True, channel=None, t_slices=None, benchmark_channel=0):

    # Call the main function with the test input
    append_scores_models = []
    for bmz_model in bmz_models:
        append_scores_data = []
        for dataset in datasets:
            input_uri, ref_uri = datasets[dataset]
            scores = process_benchmark(
                bmz_model=bmz_model,
                ome_zarr_uri=input_uri,
                reference_annotations=ref_uri,
                plot_images=plot_images,  
                z_slices=z_planes,
                crop_image=crop_image,
                channel=channel, 
                t_slices=t_slices, 
                benchmark_channel=benchmark_channel
            )
            id_array = [['Model: '+ bmz_model, 'Model: '+ bmz_model,'Model: '+ bmz_model, 'Model: '+ bmz_model, 'Model: '+ bmz_model, 'Model: '+ bmz_model, 'Model: '+ bmz_model],
                        ['Precision','Recall','IoU', 'Dice', 'PSNR','NRMSE','SSIM']]
            index = pd.MultiIndex.from_arrays(id_array, names=('Model', 'Score'))
            df = pd.DataFrame({'Dataset: '+ dataset: scores}, index=index)
            append_scores_data.append(df)
        df1 =  pd.concat(append_scores_data, axis=1)  
        append_scores_models.append(df1) 
    df2 =  pd.concat(append_scores_models)   
    return df2
