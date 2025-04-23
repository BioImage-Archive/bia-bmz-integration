import os
import json
from dataclasses import asdict
from bia_bmz_integrator.data_models.model_dataset_table import ModelDatasetTable


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

   image_filename = (
      "/bioimage-archive/ai-galleries/example-images/" +
      "example_image_" + 
      study_acc + "_" + 
      dataset_uuid + 
      ".png"
   )
   prediction_filename = (
      "/bioimage-archive/ai-galleries/example-images/" + 
      "prediction_image_" +
      study_acc + "_" + 
      dataset_uuid + "_" + 
      bmz_model + 
      ".png"
   )

   result.example_image = image_filename
   result.example_process_image = prediction_filename
   
   if scores:
      ground_truth_filename = (
         "/bioimage-archive/ai-galleries/example-images/" +
         "ground_truth_" + 
         study_acc + "_" + 
         annotation_dataset_uuid + 
         ".png"
      )
      result.example_ground_truth=ground_truth_filename
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
      bmz_model 
):
   
   output_dir = "./results/jsons"
   os.makedirs(output_dir, exist_ok=True)

   # None values as default for these are useful for the model-dataset table, 
   # but need to be a string for filenames, and may as well be something other than none 
   if study_acc is None:
      study_acc = "unspecified"
   if dataset_uuid is None:
      dataset_uuid = "unspecified"
   if annotation_dataset_uuid is None:
      annotation_dataset_uuid = "unspecified"
   
   results_list = [asdict(result)]
   output_file = (
      output_dir + 
      "/result_" + 
      study_acc + "_" + 
      dataset_uuid + "_" + 
      annotation_dataset_uuid + "_" +
      bmz_model + 
      ".json"
   )
   with open(output_file, "w") as f:
      json.dump(results_list, f, indent=4)
