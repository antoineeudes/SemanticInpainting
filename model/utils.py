import os
import zipfile
import glob
from google.cloud import storage


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print("Blob {} downloaded to {}.".format(source_blob_name, destination_file_name))


def upload_single_file_to_gcs(local_file, bucket, remote_path):
    blob = bucket.blob(remote_path)
    blob.upload_from_filename(local_file)
    print("Blob {} uploaded to {}.".format(local_file, bucket))


def upload_local_directory_to_gcs(local_path, bucket, gcs_path):
    assert os.path.isdir(local_path)
    for local_file in glob.glob(local_path + "/**"):
        if not os.path.isfile(local_file):
            upload_local_directory_to_gcs(
                local_file, bucket, gcs_path + "/" + os.path.basename(local_file)
            )
        else:
            remote_path = os.path.join(gcs_path, local_file[1 + len(local_path) :])
            upload_single_file_to_gcs(local_file, bucket, remote_path)


def download_dataset():
    # if not os.path.isdir("dataset"):
    #     os.mkdir("dataset")

    # download the dataset from Google Cloud Storage
    download_blob("semantic_inpainting", "dataset.zip", "./dataset.zip")

    # unzip the dataset
    with zipfile.ZipFile("./dataset.zip", "r") as zip_ref:
        zip_ref.extractall("./")

    # remove the zip file
    os.remove("./dataset.zip")