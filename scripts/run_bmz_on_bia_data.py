import numpy as np
import matplotlib.pyplot as plt
import click
import zarr
import dask.array as da
import bioimageio.core
from bioimageio.core import Tensor, Sample, test_model, create_prediction_pipeline
from bioimageio.spec.utils import load_array
from bioimageio.spec.model import v0_5, v0_4

# Function to create appropiate model inputs from zarr

def remote_zarr_to_model_input(ome_zarr_uri):
    """Makes many bad assumptions about shape."""
    
    zgroup = zarr.open(ome_zarr_uri)
    zarray = zgroup['0']    

    darray = da.from_zarr(zarray)

    return darray

# Function to display input and output images
def show_images(sample_tensor, prediction_tensor, inp_id, outp_id):
    input_array = np.asarray(sample_tensor.members[inp_id].data)
    input_dim= sample_tensor.members[inp_id].dims
    # select only x-y data
    x_index = input_dim.index('x')
    y_index = input_dim.index('y') 
    permutation = [y_index, x_index] + [i for i in range(len(input_dim)) if i not in [x_index, y_index]]     
    transposed_inarray = input_array.transpose(permutation)
    if transposed_inarray.ndim ==4:
        input_array=transposed_inarray[:,:,0,0]
    elif transposed_inarray.ndim ==5:
        input_array=transposed_inarray[:,:,0,0,0]

    output_array = np.asarray(prediction_tensor.members[outp_id].data)
    outp_dim= prediction_tensor.members[outp_id].dims
    # select only x-y data
    x_index = outp_dim.index('x')
    y_index = outp_dim.index('y') 
    permutation = [y_index, x_index] + [i for i in range(len(outp_dim)) if i not in [x_index, y_index]]     
    transposed_outarray = output_array.transpose(permutation)
    if transposed_outarray.ndim ==4:
        output_array=transposed_outarray[:,:,0,0]
    elif transposed_outarray.ndim ==5:
        output_array=transposed_outarray[:,:,0,0,0]


    plt.figure()
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title("Input")
    ax1.axis("off")
    plt.imshow(input_array)
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title("Prediction")
    ax2.axis("off")
    plt.imshow(output_array)
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
    

@click.command()
@click.argument("bmz_model")
@click.argument("ome_zarr_uri")
@click.option("-c", "--crop_image", nargs=2, type= int,
              help="crop image to obtain an image with the size specified. First value is x second is y]")
@click.option("-z", "--z_slices", nargs=2, type= int,
              help="select a range of z planes")
@click.option("-ch", "--channel", type= int,
              help="select a channel")
@click.option("-t", "--t_slices", nargs=2, type= int,
              help="select a range of time points")
@click.option("-p", "--plot_images", default=True,
              help="show input and output images; defaults to showing the images")

def main(bmz_model,ome_zarr_uri,plot_images,crop_image, z_slices,channel,t_slices):
    # load image
    dask_array = remote_zarr_to_model_input(ome_zarr_uri)
    # crop image if needed
    if crop_image:
        dask_array=crop_center(dask_array,crop_image)
    # select planes and channels of interest
    if z_slices:
        dask_array=dask_array[:,:,z_slices[0]:z_slices[1],:,:]
        if dask_array.ndim < 5:
            dask_array = np.expand_dims(dask_array, 2)
    if channel is not None:
        dask_array=dask_array[:,channel,:,:,:]
        dask_array = np.expand_dims(dask_array, 1)
    if t_slices:
        dask_array=dask_array[t_slices[0]:t_slices[1],:,:,:,:]
    
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

    # show images
    if plot_images:
        show_images(sample, prediction, inp_id, outp_id)

    # return output as numpy array
    output_array = np.asarray(prediction.members[outp_id].data)
    print(output_array.shape)
    return(output_array)

if __name__ == "__main__":
    main()