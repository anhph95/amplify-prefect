import base64
import os

from prefect import task
from storage.mediastore import MediaStore



@task()
def upload(image_path: str, key: str):
    """
    Upload the image at the given path to the media store using the given key.
    """
    with MediaStore(
        os.getenv("MEDIASTORE_URL"), token=os.getenv("MEDIASTORE_TOKEN")
    ) as store:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            store.put(key, encoded_string)
