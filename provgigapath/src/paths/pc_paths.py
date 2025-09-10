import os
import pathlib
from pathlib import Path


data_path1 = Path("../../../BraTS-Path/New-Data-384-Collated-JPG")
data_path2 = Path("../../../BraTS-Path/BraTS-Path2025-Train-2-JPG/Validation-Data-384-Collated-JPG")
additional_raw_data_path = data_path2
raw_data_path = data_path1
parsed_data_path = data_path1 / "Parsed"
csv_path = Path('../../../BraTS-Path/NEW_CSV')


project_path = Path("../../model")
checkpoints_path = Path("../../model/phase_7_Fulldata100/checkpoints")
logs_path = Path("../../model/phase_7_Fulldata100/logs")
figures_path = Path("../../model/phase_7_Fulldata100/figures")
models_path = project_path 
results_path = project_path / "phase_7_Fulldata100/Results"

provgigapath_path = models_path / "provgigapath_pretrained.bin"
