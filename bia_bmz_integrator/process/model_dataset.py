import os
import json
import uuid
from dataclasses import asdict
from bia_bmz_integrator.data_models.model_dataset_table import ModelDatasetTable


def get_analysis_parameters_string(analysis_parameters) -> str:

   params_dict = asdict(analysis_parameters)
   return ",".join(f"{key}={value}" for key, value in params_dict.items())


def generate_unqiue_filename(
   bmz_model, 
   study_acc, 
   dataset_uuid, 
   annotation_dataset_uuid, 
   analysis_parameters, 
   file_type,
) -> str:

   # namespace uuid generated with uuid4
   model_dataset_uuid_namespace = uuid.UUID("d7ce43f8-ad2c-4d48-ab1a-3040454ad015")

   analysis_params = get_analysis_parameters_string(analysis_parameters)
   unique_string = (
      f"{bmz_model},"
      f"{study_acc},"
      f"{dataset_uuid},"
      f"{annotation_dataset_uuid},"
      f"{analysis_params},"
      f"{file_type}"
   )

   return str(uuid.uuid5(model_dataset_uuid_namespace, unique_string))


def make_model_dataset_table(
   bmz_model, 
   study_acc, 
   dataset_uuid, 
   annotation_dataset_uuid, 
   scores, 
   analysis_parameters, 
) -> ModelDatasetTable:
   
   result = ModelDatasetTable(
      model = bmz_model,
      study = study_acc,
      dataset_uuid = dataset_uuid,
      annotation_data_set_uuid = annotation_dataset_uuid, 
      analysis_parameters = analysis_parameters
   )

   # None values as default for these are useful for the result, 
   # but need to be a string for filenames, and may as well be something other than none 
   if study_acc is None:
      study_acc = "unspecified"
   if dataset_uuid is None:
      dataset_uuid = "unspecified"
   if annotation_dataset_uuid is None:
      annotation_dataset_uuid = "unspecified"

   image_filename = generate_unqiue_filename(
      bmz_model, 
      study_acc, 
      dataset_uuid, 
      annotation_dataset_uuid, 
      analysis_parameters, 
      file_type="input_image",
   )
   prediction_filename = generate_unqiue_filename(
      bmz_model, 
      study_acc, 
      dataset_uuid, 
      annotation_dataset_uuid, 
      analysis_parameters, 
      file_type="prediction_image",
   )

   image_filepath = (
      "/bioimage-archive/ai-galleries/example-images/" +
      image_filename + 
      ".png"
   )
   prediction_filepath = (
      "/bioimage-archive/ai-galleries/example-images/" + 
      prediction_filename + 
      ".png"
   )

   result.example_image = image_filepath
   result.example_process_image = prediction_filepath
   
   if scores:
      ground_truth_filename = generate_unqiue_filename(
         bmz_model, 
         study_acc, 
         dataset_uuid, 
         annotation_dataset_uuid, 
         analysis_parameters, 
         file_type="ground_truth_image",
      )
      ground_truth_filepath = (
         "/bioimage-archive/ai-galleries/example-images/" +
         ground_truth_filename + 
         ".png"
      )
      result.example_ground_truth=ground_truth_filepath
      result.precision=scores[0] 
      result.recall=scores[1] 
      result.IoU=scores[2] 
      result.dice=scores[3]
   
   return result


def save_model_dataset_table(
      result, 
      study_acc, 
      dataset_uuid, 
      annotation_dataset_uuid, 
      bmz_model, 
      analysis_parameters,  
):
   
   output_dir = "./results/jsons/"
   os.makedirs(output_dir, exist_ok=True)

   # None values as default for these are useful for the model-dataset table, 
   # but need to be a string for filenames, and may as well be something other than none 
   if study_acc is None:
      study_acc = "unspecified"
   if dataset_uuid is None:
      dataset_uuid = "unspecified"
   if annotation_dataset_uuid is None:
      annotation_dataset_uuid = "unspecified"
   
   json_filename = generate_unqiue_filename(
      bmz_model, 
      study_acc, 
      dataset_uuid, 
      annotation_dataset_uuid, 
      analysis_parameters, 
      file_type="json_result",
   )
   
   results_list = [asdict(result)]
   output_file = (
      output_dir + 
      json_filename + 
      ".json"
   )
   with open(output_file, "w") as f:
      json.dump(results_list, f, indent=4)
