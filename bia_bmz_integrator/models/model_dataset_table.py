from dataclasses import dataclass
from typing import List

@dataclass
class AnalysisParameters:
    z_slices_analysed: list=None
    xy_image_crop_size: list=None
    input_channel_analysed: int=None
    t_slices_analysed: list=None
    prediction_channel: int=None

@dataclass
class ModelDatasetTable:
    model: str
    study: str
    dataset_uuid: str
    annotation_data_set_uuid: str
    example_image: str
    example_process_image: str
    analysis_parameters: AnalysisParameters
    example_ground_truth: str=None
    precision: float=None
    recall: float=None
    IoU: float=None
    dice: float=None
