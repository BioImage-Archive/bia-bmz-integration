import click
import json
import os
import matplotlib.pyplot as plt
from dataclasses import asdict
from bia_bmz_integrator.process.run_benchmark_models import process_benchmark, process_run
from bia_bmz_integrator.models.model_dataset_table import AnalysisParameters, ModelDatasetTable


@click.command()
@click.argument("bmz_model")
@click.argument("ome_zarr_uri")
@click.argument("reference_annotations", required=False)
@click.option("-c", "--crop_image", nargs=2, type= int,
              help="crop the input image to obtain an image with the size specified. First value is x second is y]")
@click.option("-z", "--z_slices", nargs=2, type= int,
              help="select a range of z planes from the input image")
@click.option("-ch", "--channel", type= int,
              help="select a channel from the input image")
@click.option("-t", "--t_slices", nargs=2, type= int,
              help="select a range of time points from the input image")
@click.option("-b_ch", "--benchmark_channel", type= int, default=0,
              help="select a channel to benchmark from the prediction")
@click.option("-acc", "--study_acc", type=str, default="unspecified", 
              help="the accession for the study")
@click.option("-uuid", "--dataset_uuid", type=str, default="unspecified", 
              help="the uuid of the dataset with which the image is associated")
@click.option("-p", "--plot_images", default=True,
              help="show input and output images; defaults to showing the images")


def cli(
   bmz_model, 
   ome_zarr_uri, 
   reference_annotations, 
   crop_image, 
   z_slices, 
   channel, 
   t_slices, 
   benchmark_channel, 
   study_acc, 
   dataset_uuid, 
   plot_images
):
   
   click.echo(
      "processing: \n"
      f"study_acc: {study_acc} \n"
      f"dataset_uuid: {dataset_uuid} \n"
      f"bmz model: {bmz_model} \n"
      f"ome zarr uri: {ome_zarr_uri} \n"
      f"reference annotations: {reference_annotations} \n"
      f"with cropping: {crop_image} \n"
      f"for z slices: {z_slices} \n"
      f"for channel: {channel} \n"
      f"for t slices: {t_slices} \n"
      f"with benchmark channel: {benchmark_channel} \n"
      f"and {plot_images} for plotting."
   )

   if reference_annotations:
      
      result = process_benchmark(
         bmz_model, 
         ome_zarr_uri, 
         reference_annotations, 
         plot_images, 
         crop_image, 
         z_slices, 
         channel, 
         t_slices, 
         benchmark_channel
      )
      scores = result["scores"]
      
   else:
      
      result = process_run(
         bmz_model, 
         ome_zarr_uri, 
         plot_images, 
         crop_image, 
         z_slices, 
         channel, 
         t_slices, 
         benchmark_channel
      )
      scores = None
   
   result_table = make_model_dataset_table(
      bmz_model, 
      crop_image, 
      z_slices, 
      channel, 
      t_slices, 
      benchmark_channel, 
      study_acc, 
      dataset_uuid, 
      scores  
   )
   
   save_result(result_table, study_acc, dataset_uuid, bmz_model)
   save_prediction_image(result["prediction"], study_acc, dataset_uuid, bmz_model)


def make_model_dataset_table(
   bmz_model, 
   crop_image, 
   z_slices, 
   channel, 
   t_slices, 
   benchmark_channel, 
   study_acc, 
   dataset_uuid, 
   scores
) -> ModelDatasetTable:
   
   analysis_parameters = AnalysisParameters()
   if z_slices:
      analysis_parameters.z_slices_analysed = z_slices
   if crop_image:
      analysis_parameters.xy_image_crop_size = crop_image
   if channel:
      analysis_parameters.input_channel_analysed = channel
   if t_slices:
      analysis_parameters.t_slices_analysed = t_slices
   if benchmark_channel:
      analysis_parameters.prediction_channel_benchmarked = benchmark_channel
   
   image_filename = (
      "/bioimage-archive/ai-benchmarking-galleries/example-images/" +
      "example_image_" + 
      study_acc + "_" + 
      dataset_uuid + 
      ".png"
   )
   prediction_filename = (
      "/bioimage-archive/ai-benchmarking-galleries/example-images/" + 
      "prediction_image_" +
      study_acc + "_" + 
      dataset_uuid + "_" + 
      bmz_model + 
      ".png"
   )

   result = ModelDatasetTable(
      model=bmz_model,
      study=study_acc,
      dataset_uuid=dataset_uuid,
      example_image=image_filename, 
      example_process_image=prediction_filename, 
      analysis_parameters=analysis_parameters
   )

   if scores:
      ground_truth_filename = (
         "/bioimage-archive/ai-benchmarking-galleries/example-images/" +
         "ground_truth_" + 
         study_acc + "_" + 
         dataset_uuid + 
         ".png"
      )
      result.example_ground_truth=ground_truth_filename
      result.precision=scores[0] 
      result.recall=scores[1] 
      result.IoU=scores[2] 
      result.dice=scores[3]
   
   return result


def save_result(
      result, 
      study_acc, 
      dataset_uuid, 
      bmz_model 
):
   
   output_dir = "./results/jsons"
   os.makedirs(output_dir, exist_ok=True)
   
   results_list = [asdict(result)]
   output_file = (
      output_dir + 
      "/result_" + 
      study_acc + "_" + 
      dataset_uuid + "_" + 
      bmz_model + 
      ".json"
   )
   with open(output_file, "w") as f:
      json.dump(results_list, f, indent=4)


def save_prediction_image(
   prediction, 
   study_acc, 
   dataset_uuid, 
   bmz_model
):
   
   output_dir = "./results/prediction_images"
   os.makedirs(output_dir, exist_ok=True)

   # saving central slice of image
   shape = prediction.shape
   centre_indices = tuple(s // 2 for s in shape[:3])
   centre_slice = prediction[centre_indices[0], centre_indices[1], centre_indices[2], :, :]
   image_filename = (
      output_dir + 
      "/prediction_image_" +
      study_acc + "_" + 
      dataset_uuid + "_" + 
      bmz_model + 
      ".png"
   )
   plt.imsave(image_filename, centre_slice, cmap="gray")
