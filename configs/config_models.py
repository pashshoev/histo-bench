from dataclasses import dataclass, asdict
import yaml

@dataclass
class VisionFeatureExtractionConfig:
    # Paths
    wsi_dir: str
    coordinates_dir: str
    features_dir: str
    wsi_meta_data_path: str

    # Processing
    device: str = "mps"
    patch_batch_size: int = 64
    num_workers: int = 2
    model_name: str = "resnet50"
    hf_token: str = ""

    log_level: str = "INFO"

    disable_progress_bar: bool = False

    @classmethod
    def from_yaml(cls, file_path: str) -> "VisionFeatureExtractionConfig":
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, file_path: str):
        with open(file_path, "w") as f:
            yaml.safe_dump(asdict(self), f)
