import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
import os
import random as rnd

# Load the pretrained model
model = models.resnet50(pretrained=True)

# Use the model object to select the desired layer
layer = model._modules.get('avgpool')

# Set model to evaluation mode
model.eval()

# Get the current working directory
current_directory = os.getcwd()

# Specify the relative path to the 'images' folder
relative_path = 'images'

# Construct the absolute path
absolute_path = os.path.join(current_directory, relative_path)

# Transformations
transforms = transforms.Compose([
  transforms.Resize(255),
  transforms.CenterCrop(224),
  transforms.ToTensor()
  ])

dataset = datasets.ImageFolder(absolute_path, transform=transforms)
dataset = list(dataset)

# Separate images and labels
C,labels = map(list, zip(*dataset)) 

def get_target_image_indices(target_images_num=5):
  """
  Generates a list of unique random indices representing target images.

  Args:
    total_images (int): Total number of images.
    target_images_num (int, optional, default=5): Number of target images to select.
  
  Returns:
    list: List of unique random indices.
  """

  target_image_indices = []
  
  while len(target_image_indices) < target_images_num:
    temp_rnd = rnd.randint(0, len(C))
    if (temp_rnd not in target_image_indices):
      target_image_indices.append(temp_rnd)

  return target_image_indices
if __name__ == "__main__":

  k = 5
  
  target_images = get_target_image_indices()
  print(f"Target images: {target_images}")
