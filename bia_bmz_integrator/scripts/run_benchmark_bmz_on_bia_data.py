import click
import json
import os
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt
from dataclasses import asdict
from bia_bmz_integrator.process.run_benchmark_models import process_benchmark, process_run
from bia_bmz_integrator.data_models.model_dataset_table import AnalysisParameters, ModelDatasetTable


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
@click.option("-ann_uuid", "--annotation_dataset_uuid", type=str, default="unspecified", 
              help="the uuid of the annotation dataset with which the image is associated")
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
   annotation_dataset_uuid, 
   plot_images
):
   
   print_update(
      study_acc, 
      dataset_uuid, 
      bmz_model, 
      ome_zarr_uri, 
      reference_annotations, 
      annotation_dataset_uuid, 
      crop_image, 
      z_slices, 
      channel, 
      t_slices, 
      benchmark_channel, 
      plot_images 
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
   
   result_table = make_model_dataset_table(
      bmz_model, 
      crop_image, 
      z_slices, 
      channel, 
      t_slices, 
      benchmark_channel, 
      study_acc, 
      dataset_uuid, 
      annotation_dataset_uuid, 
      result["scores"]  
   )
   
   save_result(result_table, study_acc, dataset_uuid, annotation_dataset_uuid, bmz_model)
   save_images(result, result_table)


def make_model_dataset_table(
   bmz_model, 
   crop_image, 
   z_slices, 
   channel, 
   t_slices, 
   benchmark_channel, 
   study_acc, 
   dataset_uuid, 
   annotation_dataset_uuid, 
   scores, 
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
      "/public/ai-galleries/example-images/" +
      "example_image_" + 
      study_acc + "_" + 
      dataset_uuid + 
      ".png"
   )
   prediction_filename = (
      "/public/ai-galleries/example-images/" + 
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
      annotation_data_set_uuid=annotation_dataset_uuid, 
      example_image=image_filename, 
      example_process_image=prediction_filename, 
      analysis_parameters=analysis_parameters
   )

   if scores:
      ground_truth_filename = (
         "/public/ai-galleries/example-images/" +
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


def save_result(
      result, 
      study_acc, 
      dataset_uuid, 
      annotation_dataset_uuid, 
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
      annotation_dataset_uuid + "_" +
      bmz_model + 
      ".json"
   )
   with open(output_file, "w") as f:
      json.dump(results_list, f, indent=4)


def save_images(
   result, 
   result_table, 
):
   
   input_image = result["input_image"]
   prediction_image = result["prediction_image"]
   ground_truth_image = result["ground_truth_image"]

   # saving central slice of each image
   shape = input_image.shape
   centre_indices = tuple(s // 2 for s in shape[:3])
   
   input_output = input_image[centre_indices[0], centre_indices[1], centre_indices[2], :, :]
   prediction_output = prediction_image[centre_indices[0], centre_indices[1], centre_indices[2], :, :]
   
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


def print_update(
   study_acc, 
   dataset_uuid, 
   bmz_model, 
   ome_zarr_uri, 
   reference_annotations, 
   annotation_dataset_uuid, 
   crop_image, 
   z_slices, 
   channel, 
   t_slices, 
   benchmark_channel, 
   plot_images 
):

   console = Console()

   table = Table(title="[bold magenta]Processing Details[/]")

   table.add_column("Parameter", style="cyan", justify="right")
   table.add_column("Value", style="yellow")

   table.add_row("study_acc", f"[magenta]{study_acc}[/]")
   table.add_row("dataset_uuid", f"[magenta]{dataset_uuid}[/]")
   table.add_row("bmz model", f"[cyan]{bmz_model}[/]")
   table.add_row("ome zarr uri", f"[blue]{ome_zarr_uri}[/]")
   table.add_row("reference annotations", f"[blue]{reference_annotations}[/]")
   table.add_row("annotation dataset uuid", f"[magenta]{annotation_dataset_uuid}[/]")
   table.add_row("cropping", str(crop_image) if crop_image is not None else "None")
   table.add_row("z slices", str(z_slices) if z_slices is not None else "None")
   table.add_row("channel", str(channel) if channel is not None else "None")
   table.add_row("t slices", str(t_slices) if t_slices is not None else "None")
   table.add_row("benchmark channel", str(benchmark_channel) if benchmark_channel is not None else "None")
   table.add_row("plotting", str(plot_images))

   console.print(table)

