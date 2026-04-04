import os
from pathlib import Path
from azure.storage.blob import ContainerClient, BlobServiceClient
import logging

logger = logging.getLogger(__name__)

# Suppress verbose azure http logging
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.core").setLevel(logging.WARNING)

def download_models_from_azure(local_model_dir: str = "setfit_models", container_name: str = "setfit-models"):
    """
    Downloads models from Azure Blob Storage.
    Connects using AZURE_MODELS_CONTAINER_URL if available, 
    otherwise falls back to AZURE_STORAGE_CONNECTION_STRING.
    """
    target_dir = Path(local_model_dir)
    
    url = os.getenv("AZURE_MODELS_CONTAINER_URL")
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    
    if url:
        logger.info("Connecting to Azure using Container URL...")
        container_client = ContainerClient.from_container_url(url)
    elif connection_string:
        logger.info("Connecting to Azure using Connection String...")
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)
    else:
        logger.info("No Azure credentials found. Skipping model download.")
        return

    logger.info(f"Checking models in Azure Container '{container_name}'...")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        blobs = [b for b in container_client.list_blobs()]
        if not blobs:
             logger.warning("No models found in the Azure container.")
             return
             
        download_count = 0
        
        logger.info("Downloading models ....")
        for blob in blobs:
            blob_client = container_client.get_blob_client(blob)
            download_file_path = target_dir / blob.name
            
            # Skip download if file already exists and size perfectly matches.
            # (Allows faster restarts if models are mounted on a volume)
            if download_file_path.exists() and download_file_path.stat().st_size == blob.size:
                continue

            download_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(download_file_path, "wb") as file:
                file.write(blob_client.download_blob().readall())
            download_count += 1
            
        if download_count > 0:
            logger.info("✅ Successfully downloaded %d updated model files from Azure.", download_count)
        else:
            logger.info("✅ Local models are already up-to-date with Azure.")
            
    except Exception as e:
        logger.error("❌ Failed to download models from Azure: %s", e)

if __name__ == "__main__":
    # Configure basic logging if executed directly
    logging.basicConfig(level=logging.INFO)
    download_models_from_azure()
