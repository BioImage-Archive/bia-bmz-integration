import click
from bia_bmz_integration import process
   

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
   return process(bmz_model,ome_zarr_uri,reference_annotations,plot_images,crop_image, z_slices,channel,t_slices, benchmark_channel)

if __name__ == "__main__":
    main()