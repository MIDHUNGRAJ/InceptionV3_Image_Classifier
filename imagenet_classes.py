from pathlib import Path
import requests

# Define paths
imgnet_path = Path("imgnet")
imgnet_classes_path = imgnet_path / "imagenet_classes.txt"

# Check if the file exists
if imgnet_classes_path.exists():
    print(f"{imgnet_classes_path} file exists.")
else:
    # If the file doesn't exist, create it
    print(f"Did not find {imgnet_classes_path} file, creating one...")
    imgnet_path.mkdir(parents=True, exist_ok=True)

    # Download the file
    with open(imgnet_classes_path, "wb") as f:
        print("Downloading file...")
        response = requests.get("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt")
        f.write(response.content)

print("Process completed.")
