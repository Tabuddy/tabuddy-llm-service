

import os
from pathlib import Path
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError

def upload_models_to_azure(local_model_dir: str = "setfit_models", container_name: str = "setfit-models"):
    """
    Recursively uploads the contents of the local_model_dir to an Azure Blob Storage container.
    """
    # Load environment variables from .env
    load_dotenv()
    
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        print("Error: AZURE_STORAGE_CONNECTION_STRING is not set in the environment.")
        print("Please add it to your .env file.")
        return

    print("Connecting to Azure Blob Storage...")
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    
    # Get or create the container
    container_client = blob_service_client.get_container_client(container_name)
    try:
        container_client.create_container()
        print(f"Created new container '{container_name}'.")
    except ResourceExistsError:
        print(f"Container '{container_name}' already exists.")

    model_path = Path(local_model_dir)
    if not model_path.exists() or not model_path.is_dir():
        print(f"Error: Local directory '{local_model_dir}' not found.")
        return

    print(f"Uploading files from '{local_model_dir}'...")
    upload_count = 0
    
    # Recursively upload files
    for filepath in model_path.rglob("*"):
        if filepath.is_file():
            # Calculate the relative path to use as the blob name
            # E.g., if filepath is setfit_models/tier1/model.safetensors
            # the blob name will be tier1/model.safetensors
            blob_name = filepath.relative_to(model_path).as_posix()
            
            blob_client = container_client.get_blob_client(blob_name)
            
            print(f"  -> Uploading {blob_name} ({filepath.stat().st_size / (1024*1024):.2f} MB)...")
            
            with open(filepath, "rb") as data:
                # Overwrite = True to ensure retrained models overwrite old ones
                blob_client.upload_blob(data, overwrite=True)
            
            upload_count += 1

    print(f"\nSuccessfully uploaded {upload_count} files to Azure Blob Storage.")
    print(f"Container: {container_name}")

if __name__ == "__main__":
    upload_models_to_azure()
