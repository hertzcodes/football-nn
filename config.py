from dataclasses import dataclass, field
from typing import List, Tuple

import torch

@dataclass
class Config:
    """
    Docstring for Config
    """
    input_video_path: str = "./input_video.mp4"
    output_video_path: str = "./output_video.mp4"

    device: str = "mps" if torch.backends.mps.is_available() else "cpu"

    colors: List[Tuple[int, int, int]] = field(
        default_factory=lambda: [
            (0, 215, 255),   # referees/coaches – amber
            (50, 205, 50),   # team 1 – green
            (60, 20, 220),   # team 2 – deep red
        ]
    )

    cache_dir: str = "cache"
    use_cache: bool = True
