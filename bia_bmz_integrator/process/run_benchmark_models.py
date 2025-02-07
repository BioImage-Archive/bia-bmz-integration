import numpy as np
import torch
import pandas as pd
import warnings
import cv2
from bia_bmz_integrator.process.image_utils import (
    remote_zarr_to_model_input, 
    crop_center, 
    slice_image, 
    reorder_array_dimensions, 
    show_images, 
    show_images_gt, 
)
from bia_bmz_integrator.process.predict import run_model_inference
from bia_bmz_integrator.process.benchmark import segmentation_scores, restoration_scores


def process_run(
    bmz_model, 
    ome_zarr_uri, 
    crop_image, 
    z_slices, 
    channel, 
    t_slices, 
    benchmark_channel, 
    plot_images
) -> dict:
    
    # Load image and annotations
    dask_array = remote_zarr_to_model_input(ome_zarr_uri)

    # slice image
    sliced_array = slice_image(dask_array, crop_image, z_slices, channel, t_slices)

    # Run model inference
    new_np_array = np.squeeze(sliced_array).compute()
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
    output["scores"] = None
    output["input_image"] = input_array
    output["prediction_image"] = correct_ch_output
    output["ground_truth_image"] = None

    return output
    

def process_benchmark(
    bmz_model, 
    ome_zarr_uri, 
    reference_annotations, 
    crop_image, 
    z_slices, 
    channel, 
    t_slices, 
    benchmark_channel, 
    plot_images
) -> dict:
    
    # Load image and annotations
    dask_array = remote_zarr_to_model_input(ome_zarr_uri)
    ref_array = remote_zarr_to_model_input(reference_annotations)

    # slice images
    sliced_array = slice_image(dask_array, crop_image, z_slices, channel, t_slices)
    ref_array = slice_image(ref_array, crop_image, z_slices, 0, t_slices)

    # Run model inference
    new_np_array = np.squeeze(sliced_array).compute()
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
    output["input_image"] = input_array
    output["prediction_image"] = correct_ch_output
    output["ground_truth_image"] = ref_array

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
