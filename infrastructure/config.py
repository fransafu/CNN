import os

from dataclasses import dataclass
from yamldataclassconfig.config import YamlDataClassConfig

from infrastructure.errors import IncorrectBasePathDir


@dataclass
class Config(YamlDataClassConfig):
    # General
    basepath_dir: str = None
    use_multithreads: bool = False
    num_threads: int = None

    # Dataset
    dataset_dir: str = None

    # Images description
    image_type: str = None
    channels: int = None
    image_width: int = None
    image_height: int = None

    # Model
    num_epochs: int = None
    num_classes: int = None
    batch_size: int = None
    validation_steps: int = None
    learning_rate: float = None
    use_l2: bool = None
    weight_decay: float = None
    decay_steps: int = None
    shuffle_size: int = None
    optimizer_name: str = None
    history_path: str = None
    use_imagenet_weights: bool = None
    imagenet_dir: str = None

    # Load pre-trained weights
    pretrained_weights: str = None

    # Snapshot
    snapshot_dir: str = None
    snapshot_steps: int = None
    checkpoint_file: str = None
    use_checkpoint: bool = False

    def isValid(self) -> bool:
        relative_path = os.path.realpath(__file__)
        relative_path = relative_path.replace('/infrastructure/config.py', '')
        if self.basepath_dir != relative_path:
            raise IncorrectBasePathDir()

    def set_dataset_dir(self, path):
        self.dataset_dir = path