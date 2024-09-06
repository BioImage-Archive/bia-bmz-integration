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
from monai.transforms import ForegroundMask
import torch

# Function to create appropiate model inputs from zarr

def remote_zarr_to_model_input(ome_zarr_uri):
    """Makes many bad assumptions about shape."""
    
    zgroup = zarr.open(ome_zarr_uri)
    zarray = zgroup['0']    

    darray = da.from_zarr(zarray)

    return darray

# Function to display input and output images
def show_images(sample_tensor, prediction_tensor, reference_annotations, inp_id, outp_id,benchmark_channel):
    input_array = np.asarray(sample_tensor.members[inp_id].data)
    input_dim= sample_tensor.members[inp_id].dims
    # select only x-y data
    x_index = input_dim.index('x')
    y_index = input_dim.index('y')
    permutation = [y_index, x_index] + [i for i in range(len(input_dim)) if i not in [x_index, y_index]]     
    transposed_inarray = input_array.transpose(permutation)
    #transposed_reference_annotations = reference_annotations.transpose(permutation)
    if transposed_inarray.ndim ==4:
        input_array=transposed_inarray[:,:,0,0]
    elif transposed_inarray.ndim ==5:
        input_array=transposed_inarray[:,:,0,0,0]
    reference_annotations= reference_annotations[0,0,0,:,:]

    output_array = np.asarray(prediction_tensor.members[outp_id].data)
    outp_dim= prediction_tensor.members[outp_id].dims
    # select channel
    bch_index = outp_dim.index('channel')
    correct_ch_output=np.take(output_array, [benchmark_channel],bch_index)
    # select only x-y data
    x_index = outp_dim.index('x')
    y_index = outp_dim.index('y') 
    xindices = range(correct_ch_output.shape[x_index])
    yindices = range(correct_ch_output.shape[y_index])
    correct_x_output=np.take(correct_ch_output, xindices,[x_index])
    output_array=np.take(correct_x_output, yindices,[y_index])
    '''
    permutation = [y_index, x_index] + [i for i in range(len(outp_dim)) if i not in [x_index, y_index]]     
    transposed_outarray = output_array.transpose(permutation)
    if transposed_outarray.ndim ==4:
        output_array=transposed_outarray[:,:,0,0]
    elif transposed_outarray.ndim ==5:
        output_array=transposed_outarray[:,:,0,0,0]
'''
    plt.figure()
    ax1 = plt.subplot(1, 3, 1)
    ax1.set_title("Input")
    ax1.axis("off")
    plt.imshow(input_array)
    ax2 = plt.subplot(1, 3, 2)
    ax2.set_title("Prediction")
    ax2.axis("off")
    plt.imshow(output_array)
    ax3 = plt.subplot(1, 3, 3)
    ax3.set_title("Reference annotations")
    ax3.axis("off")
    plt.imshow(reference_annotations)
    plt.show()

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
    print(in_dim)
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
    print(f"input shape: {dask_array.shape}")
    print(f"input type: {dask_array.dtype}")
    # load reference annotations
    ref_array = remote_zarr_to_model_input(reference_annotations)
    print(f"reference shape: {ref_array.shape}")
    print(f"reference type: {ref_array.dtype}")
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
        print(f"test input type: {test_input_image.dtype}")
        print(f"test input shape: {test_input_image.shape}")
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
        print(f"test input type: {test_input_image.dtype}")
        print(f"test input shape: {test_input_image.shape}")
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

    print(f"sample shape: {sample.members[inp_id].tagged_shape}")
    print(f"sample type: {sample.members[inp_id].dtype}")
    # create prediction pipeline

    devices = None
    weight_format = None

    prediction_pipeline = create_prediction_pipeline(
        model_resource, devices=devices, weight_format=weight_format
    )

    # run prediction

    prediction: Sample = prediction_pipeline.predict_sample_without_blocking(sample)
    print(f"prediction shape: {prediction.members[outp_id].tagged_shape}")
    print(f"prediction type: {prediction.members[outp_id].dtype}")

    # reorder prediction dimensions to "TCZYX" so metrics can be computed 
    output_array = reorder_array_dimensions(prediction,outp_id)
    input_array = reorder_array_dimensions(sample,inp_id)

    # select correct channel to benchmark from prediction
    correct_ch_output=np.take(output_array, [benchmark_channel],1)
    print(correct_ch_output.shape)


    #output_tensor= torch.from_numpy(output_array)

    #thres_tensor = ForegroundMask(threshold = 0.95)
    #binary_output = thres_tensor(prediction.members[outp_id])
    #print(f" binary prediction shape: {binary_output.tagged_shape}")

    #indices = torch.tensor([benchmark_channel])
    #output=torch.index_select(output_tensor, bch_index, indices)
    #print(f"new prediction shape: {torch.index_select(prediction.members[outp_id], bch_index, indices).tagged_shape}")


    
    #output_array = output_array[:,benchmark_channel,:,:,:]
    #output_array = np.expand_dims(output_array, 1)

    # create tendor with prediction and binarize prediction to calculate metrics 
    t_output = torch.from_numpy(correct_ch_output)
    binary_output = correct_ch_output >= 0.5 #0.95
    t_binary_output = torch.from_numpy(binary_output)
    t_input = torch.from_numpy(input_array)

    # create tensor with reference and binarize reference annotation 
    ref_array = np.asarray(ref_array)
    print(f"ref min: {np.amin(ref_array)}")
    print(f"ref max: {np.amax(ref_array)}")
    print(f"out min: {np.amin(correct_ch_output)}")
    print(f"out max: {np.amax(correct_ch_output)}")
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