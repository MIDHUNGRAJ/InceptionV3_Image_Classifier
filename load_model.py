from torchvision import transforms
import torch
from pathlib import Path
from PIL import Image
import torchvision
from imagenet_classes import imgnet_classes_path


# Check if CUDA is available, else use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Inception v3 model pretrained on ImageNet dataset
# model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
model = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.DEFAULT)
model.eval()

class PrePro():
    """
    This class performs all necessary preprocessing for user input images and provides probabilities using a pre-trained model.
    """
    def preprocess(self, img, device=device):
        """
        This function takes input data, processes it, and returns the processed data.

        Parameters:
        - img (numpy.ndarray): The input image data.
        - device (str): The device to which the processed data will be moved. Defaults to the value of the 'device' variable.

        Returns:
        - input_batch (torch.Tensor): The processed image data as a batch tensor, ready for model input.
        """

        # Convert the input image data to a PIL Image
        input_image = Image.fromarray(img)
        
        # Apply transformations to the input image to prepare it for the model
        input_tensor = self.transform_img(input_image)
        
        # Add a batch dimension to the input tensor and move it to the specified device
        input_batch = input_tensor.unsqueeze(0).to(device)
        
        # Return the processed image data
        return input_batch

    
    def transform_img(self, img):
        """
        This function takes input data, transforms it, and returns the transformed data.

        Parameters:
        - img (PIL.Image): The input image data to be transformed.

        Returns:
        - transformed (torch.Tensor): The transformed image data.
        """

        # Define the sequence of transformations to be applied to the input image
        transform = transforms.Compose([
            transforms.Resize(299),  # Resize the image to 299x299 pixels
            transforms.CenterCrop(299),  # Crop the image to 299x299 pixels around the center
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image
        ])

        # Apply the defined transformations to the input image
        transformed = transform(img)

        # Return the transformed image data
        return transformed


    def predict(self, data):
        """
        This function takes input data, feeds it into the model, and returns the model's output.

        Parameters:
        - data (tensor): The input data to be fed into the model.

        Returns:
        - output (tensor): The output tensor produced by the model.
        """
        # Ensure that no gradients are calculated during inference
        with torch.no_grad():
            # Move the model to the appropriate device (CPU or GPU)
            model.to(device)
            # Feed the input data into the model and get the output
            output = model(data)

        # Return the model's output
        return output

    
    def probability(self, output):
        """
        This function calculates the probabilities for each category based on the model output.

        Parameters:
        - output (tensor): The output tensor from the model.

        Returns:
        - results (dict): A dictionary containing the top 5 categories with their corresponding probabilities.
        """
        # Calculate probabilities
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Read categories from file
        with open(imgnet_classes_path, "r") as f:
            categories = [s.strip() for s in f.readlines()]

        # Take top 5 probabilities and their corresponding category IDs
        top5_prob, top5_catid = torch.topk(probabilities, 5)

        # Initialize dictionary to store results
        results = {}

        # Loop through the top 5 probabilities
        for i in range(top5_prob.size(0)):
            # Assign category and its corresponding probability to the dictionary
            results[categories[top5_catid[i]]] = top5_prob[i].item()

        return results


