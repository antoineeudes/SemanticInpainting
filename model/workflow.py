from train_gan import train, get_arguments
import torch
from utils import download_blob, upload_local_directory_to_gcs, download_dataset
from google.cloud import storage


if __name__ == "__main__":
    args = get_arguments()
    print("on GPU:", torch.cuda.is_available())
    print(args)
    print("downloading dataset...")
    download_dataset()
    print("dataset downloaded")

    print("training dataset...")
    train(args)
    print("training done")

    print("uploading results...")
    storage_client = storage.Client()
    bucket = storage_client.bucket("semantic_inpainting")
    upload_local_directory_to_gcs("./checkpoints", bucket, "checkpoints")
    print("results uploaded")
