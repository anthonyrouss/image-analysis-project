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
