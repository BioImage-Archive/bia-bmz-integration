import numpy as np
import bioimageio.core
from bioimageio.core import Tensor, Sample, create_prediction_pipeline
from bioimageio.spec.utils import load_array
from bioimageio.spec.model import v0_5, v0_4


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
