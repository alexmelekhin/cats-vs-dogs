import os
import zipfile
from os import PathLike
from pathlib import Path
from typing import Union


def load_and_unzip(url: str, directory: Union[str, PathLike]) -> None:
    directory = Path(directory)
    if not directory.exists():
        directory.mkdir(parents=True)
    zip_file_name = directory / Path(url).name
    print(f"Downloading {url} to {zip_file_name}...")
    os.system(f"wget --secure-protocol=TLSv1_2 {url} -O {zip_file_name}")
    print(f"Unzipping {zip_file_name}...")
    with zipfile.ZipFile(zip_file_name, "r") as zip_ref:
        zip_ref.extractall(directory)
    os.remove(zip_file_name)
