import torch
import torchvision.transforms as transforms
from model import load_model
from PIL import Image
import numpy as np
import PIL.ImageOps
from model import Network  # Import your custom network (assuming it's in network.py)


# Load the trained model
model = Network()
model.load_state_dict(torch.load('network.pth'))  # Load your saved model file

# Set the model to evaluation mode
model.eval()

# # Define the transformation for preprocessing the custom image
# transform = transforms.Compose([
#     transforms.Resize((28, 28)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

# # Function to predict on a custom image
# def predict_custom_image(custom_image_path):
#     try:
#         # Open and preprocess the custom image
#         custom_image = Image.open(custom_image_path).convert('L')
#         custom_image = transform(custom_image)
#         custom_image = custom_image.unsqueeze(0)  # Add batch dimension

#         # Make predictions using your model
#         with torch.no_grad():
#             result = model(custom_image)

#         # Process and return predictions
#         predicted_class = torch.argmax(result)
#         return predicted_class.item()

#     except Exception as e:
#         print(f"Error: {str(e)}")
#         return None

# # Example usage
# if __name__ == '__main__':
#     custom_image_path = 'test.png'  # Correct file name and extension
#     predicted_class = predict_custom_image(custom_image_path)

#     if predicted_class is not None:
#         print(f'Predicted Class: {predicted_class}')
#     else:
#         print("Failed to predict. Check the image file and code.")


# Look image processing
from PIL import Image
import numpy as np
import PIL.ImageOps   
import matplotlib.pyplot as plt
import torch
from model import Network  # Import your custom network (assuming it's in network.py)
import os
# Load the trained model
model = Network()
model.load_state_dict(torch.load('network.pth'))  # Load your saved model file
model.eval()

# Load and preprocess the custom image
# script_directory = os.path.dirname(__file__)  # Get the directory where the script is located
# file_path = os.path.join(script_directory, 'firstTest.png')  # Replace 'your_file.jpg' with your actual filename
# print(file_path)

img = Image.open("/Users/steventussel/playingWithPandas/MNSITplayingWfire/eight.png")
img = img.resize((28, 28))
img = img.convert("L")
img = PIL.ImageOps.invert(img)

# Display the preprocessed image using matplotlib
plt.imshow(img)
plt.show()

# Convert the image to a NumPy array
img = np.array(img)
img = img / 255
image = torch.from_numpy(img)
image = image.float()

# Make predictions using your model
with torch.no_grad():
    result = model(image.view(-1, 28 * 28))

# Print the predicted class
predicted_class = torch.argmax(result)
print(f'Predicted Class: {predicted_class.item()}')
