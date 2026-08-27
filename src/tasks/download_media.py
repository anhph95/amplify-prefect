import base64
import os
from pathlib import Path

from prefect import task
from storage.mediastore import MediaStore


@task()
def download(key: str, tmp_dir: str):
    """
    Download the media item with the given key from the media store, and store it in the given temporary directory.
    """
    with MediaStore(
        os.getenv("MEDIASTORE_URL"), token=os.getenv("MEDIASTORE_TOKEN")
    ) as store:
        data = base64.b64decode(store.get(key))
        with open(Path(tmp_dir) / key, "wb") as image_file:
            image_file.write(data)
