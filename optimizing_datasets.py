from pathlib import Path

from ultralytics.data.utils import compress_one_image
from ultralytics.utils.downloads import zip_directory


path = Path("datasets/pokemon")
for f in path.rglob("*.jpg"):
    compress_one_image(f)


zip_directory("datasets/")