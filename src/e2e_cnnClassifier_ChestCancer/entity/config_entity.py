from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    unzip_dir: Path
    
@dataclass(frozen=True)
class BaseModelConfigAndParams:
    base_model_root_dir: str
    base_model_path: str
    updated_model_path: str
    
    model_name: str
    model_input_shape: List[int]
    model_num_classes: int
    
    data_train_data_dir: str
    data_batch_size: int
    data_image_size: List[int]
    
    training_weights: str
    training_include_top: bool
    training_epochs: int
    training_learning_rate: float