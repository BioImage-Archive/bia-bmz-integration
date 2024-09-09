import numpy as np
import matplotlib.pyplot as plt
import click
import zarr
import dask.array as da
import bioimageio.core
from bioimageio.core import Tensor, Sample, test_model, create_prediction_pipeline
from bioimageio.spec.utils import load_array
from bioimageio.spec.model import v0_5, v0_4
from monai.metrics import DiceMetric, MeanIoU, PSNRMetric, RMSEMetric, SSIMMetric
import torch

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
def reorder_array_dimensions(in_tensor,in_id):

    in_array = np.asarray(in_tensor.members[in_id].data)
    in_dim = in_tensor.members[in_id].dims
    bch_index = in_dim.index('channel')
    x_index = in_dim.index('x')
    y_index = in_dim.index('y') 

    if 'z' in in_dim:
        z_index = in_dim.index('z') 
        array_tczxy = np.transpose(in_array, (0, bch_index,z_index, y_index,x_index))
    else:
        array_tczxy = np.transpose(in_array, (0, bch_index, y_index,x_index))
        array_tczxy = np.expand_dims(array_tczxy, 2)

    return array_tczxy   

@click.command()
@click.argument("bmz_model")
@click.argument("ome_zarr_uri")
@click.argument("reference_annotations")
@click.option("-c", "--crop_image", nargs=2, type= int,
              help="crop the input image to obtain an image with the size specified. First value is x second is y]")
@click.option("-z", "--z_slices", nargs=2, type= int,
              help="select a range of z planes from the input image")
@click.option("-ch", "--channel", type= int,
              help="select a channel from the input image")
@click.option("-t", "--t_slices", nargs=2, type= int,
              help="select a range of time points from the input image")
@click.option("-p", "--plot_images", default=True,
              help="show input and output images; defaults to showing the images")
@click.option("-b_ch", "--benchmark_channel", type= int, default=0,
              help="select a channel to benchmark from the prediction")

def main(bmz_model,ome_zarr_uri,reference_annotations,plot_images,crop_image, z_slices,channel,t_slices, benchmark_channel):
    # load image
    dask_array = remote_zarr_to_model_input(ome_zarr_uri)

    # load reference annotations
    ref_array = remote_zarr_to_model_input(reference_annotations)

    # crop image if needed
    if crop_image:
        dask_array=crop_center(dask_array,crop_image)
        ref_array=crop_center(ref_array,crop_image)
    # select planes and channels of interest
    if z_slices:
        dask_array = dask_array[:,:,z_slices[0]:z_slices[1],:,:]
        ref_array = ref_array[:,:,z_slices[0]:z_slices[1],:,:]
        if dask_array.ndim < 5:
            dask_array = np.expand_dims(dask_array, 2)
            ref_array = np.expand_dims(ref_array, 2)
    if channel is not None:
        dask_array = dask_array[:,channel,:,:,:]
        dask_array = np.expand_dims(dask_array, 1)
        ref_array = ref_array[:,channel,:,:,:]
        ref_array = np.expand_dims(ref_array, 1)
    if t_slices:
        dask_array = dask_array[t_slices[0]:t_slices[1],:,:,:,:]
        ref_array = ref_array[t_slices[0]:t_slices[1],:,:,:,:]
    
    # load model
    model_resource = bioimageio.core.load_description(bmz_model)

    # test model

    # test_summary = test_model(model_resource)
    # print(test_summary)

    # check model specification version
    if isinstance(model_resource, v0_5.ModelDescr):
        # load test image
        test_input_image = load_array(model_resource.inputs[0].test_tensor)

        # match test data type with the data type of the model input
        dask_array = dask_array.astype(test_input_image.dtype)

        # reshape data to match model input dimensions
        new_np_array=np.squeeze(dask_array).compute()

        indices = [i for i in range(len(test_input_image.shape)) if test_input_image.shape[i] == 1]
        right_dims = np.expand_dims(new_np_array, indices)

        # convert image to tensor
        input_tensor = Tensor.from_numpy(right_dims, dims=tuple(model_resource.inputs[0].axes))

        # create collection of tensors (sample)
        inp_id=model_resource.inputs[0].id
        outp_id=model_resource.outputs[0].id
        sample = Sample(members={inp_id: input_tensor}, stat={}, id="id")

    elif isinstance(model_resource, v0_4.ModelDescr):
        # load test image
        test_input_image = load_array(model_resource.test_inputs[0])

        # match test data type with the data type of the model input
        dask_array = dask_array.astype(test_input_image.dtype)

        # reshape data to match model input dimensions
        new_np_array=np.squeeze(dask_array).compute()

        indices = [i for i in range(len(test_input_image.shape)) if test_input_image.shape[i] == 1]
        right_dims = np.expand_dims(new_np_array, indices)

        # convert image to tensor 
        input_tensor = Tensor.from_numpy(right_dims, dims=tuple(model_resource.inputs[0].axes))

        # create collection of tensors (sample)
        inp_id=model_resource.inputs[0].name
        outp_id=model_resource.outputs[0].name
        sample = Sample(members={inp_id: input_tensor}, stat={}, id="id")

    else:
        print("This model specification version is not supported")

    # create prediction pipeline

    devices = None
    weight_format = None

    prediction_pipeline = create_prediction_pipeline(
        model_resource, devices=devices, weight_format=weight_format
    )

    # run prediction

    prediction: Sample = prediction_pipeline.predict_sample_without_blocking(sample)

    # reorder prediction dimensions to "TCZYX" so metrics can be computed 
    output_array = reorder_array_dimensions(prediction,outp_id)
    input_array = reorder_array_dimensions(sample,inp_id)

    # select correct channel to benchmark from prediction
    correct_ch_output=np.take(output_array, [benchmark_channel],1)

    # create prediction torch tensor and binarize prediction to calculate metrics 
    t_output = torch.from_numpy(correct_ch_output)
    binary_output = correct_ch_output >= 0.5 #0.95
    t_binary_output = torch.from_numpy(binary_output)
    t_input = torch.from_numpy(input_array)

    # create reference torch tensor and binarize reference annotation 
    ref_array = np.asarray(ref_array)
    ref_array = ref_array.astype(test_input_image.dtype)
    t_reference = torch.from_numpy(ref_array)
    (t_reference.shape)
    ref_array_binary = ref_array.astype(bool)
    t_reference_binary = torch.from_numpy(ref_array_binary)

    # compute benchamarking metrics

    # segmentation
    iou_metric = MeanIoU(include_background=True, reduction="mean")
    iou_score=iou_metric(y_pred=t_binary_output, y=t_reference_binary)
    print(f"IoU score: {iou_score}")

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_score = dice_metric(y_pred=t_binary_output, y=t_reference_binary)
    print(f"Dice score: {dice_score}")
    
    # reconstruction
    max_val = np.max(ref_array)-np.min(ref_array)
    
    psnr_metric = PSNRMetric(max_val, reduction="mean", get_not_nans=False)
    psnr_score = psnr_metric(y_pred=t_output, y=t_reference)
    print(f"PSNR score reconstructed image: {psnr_score}")
    psnr_score = psnr_metric(y_pred=t_input, y=t_reference)
    print(f"PSNR score input image: {psnr_score}")
    
    rmse_metric = RMSEMetric(reduction="mean", get_not_nans=False)
    rmse_score = rmse_metric(y_pred=t_output, y=t_reference)
    print(f"RMSE score reconstructed image: {rmse_score}")
    rmse_score = rmse_metric(y_pred=t_input, y=t_reference)
    print(f"RMSE score input image: {rmse_score}")

    ssim_metric = SSIMMetric(spatial_dims = 2, data_range=max_val)
    ssim_score = ssim_metric(y_pred=t_output[0,:,:,:,:], y=t_reference[0,:,:,:,:])
    print(f"SSIM score reconstructed image: {ssim_score}")
    ssim_score = ssim_metric(y_pred=t_input[0,:,:,:,:], y=t_reference[0,:,:,:,:])
    print(f"SSIM score input image: {ssim_score}")


    if plot_images:
        plt.figure()
        ax1 = plt.subplot(2, 3, 1)
        ax1.set_title("Input")
        ax1.axis("off")
        plt.imshow(dask_array[0,0,0,:,:])
        ax2 = plt.subplot(2, 3, 2)
        ax2.set_title("Prediction")
        ax2.axis("off")
        plt.imshow(correct_ch_output[0,0,0,:,:])
        ax3 = plt.subplot(2, 3, 3)
        ax3.set_title("Reference annotation")
        ax3.axis("off")
        plt.imshow(ref_array[0,0,0,:,:])
        ax4 = plt.subplot(2, 3, 4)
        ax4.set_title("Reference annotation binary")
        ax4.axis("off")
        plt.imshow(ref_array_binary[0,0,0,:,:])
        ax5 = plt.subplot(2, 3, 5)
        ax5.set_title("Prediction binary")
        ax5.axis("off")
        plt.imshow(binary_output[0,0,0,:,:])
        plt.show()


if __name__ == "__main__":
    main()