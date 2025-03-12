import numpy as np
from numpy.typing import NDArray
from typing import Any
import matplotlib.pyplot as plt
import zarr
from PIL import Image, ImageOps
import dask.array as da
import os


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


def scale_to_uint8(
    image: NDArray[Any], 
) -> NDArray[Any]:

    scaled = image.astype(np.float32)

    if scaled.max() - scaled.min() == 0:
        return np.zeros(image.shape, dtype=np.uint8)

    scaled = 255 * (scaled - scaled.min()) / (scaled.max() - scaled.min())

    return scaled.astype(np.uint8)


def convert_to_greyscale(
    image: NDArray[Any], 
) -> Image.Image:
   
   image = Image.fromarray(image)

   if image.mode != "L":
      image = image.convert("L")
   
   return image


def adjust_image_brightness(
    image_array: NDArray[Any], 
    correction_type: str, 
) -> Image.Image:
    
    image = Image.fromarray(image_array)

    if correction_type == "auto":
        # the (0, 1) leaves top 1% out of histogram stretch
        image = ImageOps.autocontrast(image, (0, 1)) 
    elif correction_type == "gamma":
        image = adjust_gamma(image)
    
    return image


def adjust_gamma(
    image: Image.Image, 
) -> Image.Image:

    # hardcoded gamma for now
    gamma = 1.5
    inv_gamma = 1.0 / gamma  
    table = [int((i / 255.0) ** inv_gamma * 255) for i in range(256)]
    
    return image.point(table)


def save_images(
   result, 
   result_table, 
   adjust_image, 
):
   
   input_image = result["input_image"]
   prediction_image = result["prediction_image"]
   ground_truth_image = result["ground_truth_image"]

   # saving central slice of each image
   shape = input_image.shape
   centre_indices = tuple(s // 2 for s in shape[:3])
   
   input_output = input_image[centre_indices[0], centre_indices[1], centre_indices[2], :, :]
   prediction_output = prediction_image[centre_indices[0], centre_indices[1], centre_indices[2], :, :]

   if adjust_image is not None:
        input_output = scale_to_uint8(input_output)
        prediction_output = scale_to_uint8(prediction_output)

        input_output = adjust_image_brightness(input_output, adjust_image) 
        prediction_output = adjust_image_brightness(prediction_output, adjust_image)
      
   input_filename = os.path.basename(result_table.example_image)
   prediction_filename = os.path.basename(result_table.example_process_image)

   output_dir = "./results/images/"
   os.makedirs(output_dir, exist_ok=True)

   plt.imsave(output_dir + input_filename, input_output, cmap="gray")
   plt.imsave(output_dir + prediction_filename, prediction_output, cmap="gray")
   
   if ground_truth_image is not None:
      ground_truth_output = ground_truth_image[centre_indices[0], centre_indices[1], centre_indices[2], :, :]
      ground_truth_filename = os.path.basename(result_table.example_ground_truth)
      plt.imsave(output_dir + ground_truth_filename, ground_truth_output, cmap="gray")

