import numpy as np
import matplotlib.pyplot as plt
import zarr
import dask.array as da


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


# to do all types of slicing
def slice_image(
    image_array, 
    crop_image, 
    z_slices, 
    channel, 
    t_slices, 
):

    if crop_image:
        image_array = crop_center(image_array, crop_image)

    if z_slices:
        image_array = image_array[:, :, z_slices[0]:z_slices[1], :, :]
        if image_array.ndim < 5:
            image_array = np.expand_dims(image_array, 2)

    # channel always has value, never None
    image_array = image_array[:, channel, :, :, :]
    image_array = np.expand_dims(image_array, 1)

    if t_slices:
        image_array = image_array[t_slices[0]:t_slices[1], :, :, :, :]

    return image_array
    

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
