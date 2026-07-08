"""Download and extract the MovieLens ml-latest-small dataset into data/raw."""
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys.utils.logger import get_logger  # noqa: E402

DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
RAW_DIR = Path("data/raw")

logger = get_logger(__name__)


def download_dataset(url: str = DATASET_URL, raw_dir: Path = RAW_DIR) -> Path:
    """Download the MovieLens zip (if missing) and extract ratings/movies CSVs."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "ml-latest-small.zip"
    extracted_dir = raw_dir / "ml-latest-small"

    if (extracted_dir / "ratings.csv").exists():
        logger.info("Dataset already present at %s", extracted_dir)
        return extracted_dir

    logger.info("Downloading MovieLens dataset from %s", url)
    urlretrieve(url, zip_path)

    logger.info("Extracting %s", zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)

    zip_path.unlink(missing_ok=True)
    logger.info("Dataset ready at %s", extracted_dir)
    return extracted_dir


if __name__ == "__main__":
    download_dataset()
