from dataclasses import dataclass
from typing import List

@dataclass
class AnalysisParameters:
    z_slices_analysed: list
    xy_image_crop_size: list
    input_channel_analysed: int
    t_slices_analysed: list
    prediction_channel: int

@dataclass
class ModelDatasetTable:
    model: str
    study: str
    dataset_uuid: str
    annotation_data_set_uuid: str
    analysis_parameters: AnalysisParameters
    example_image: str=None
    example_process_image: str=None
    example_ground_truth: str=None
    precision: float=None
    recall: float=None
    IoU: float=None
    dice: float=None
