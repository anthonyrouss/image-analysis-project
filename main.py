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

def get_vector(image_path):
    """
    Extracts a feature vector from the given image using a pre-trained ResNet50 model.

    Args:
      image_path (str): The file path of the input image.
    
    Returns:
      torch.Tensor: The feature vector extracted from the image.
    """
    
    # Load the image with Pillow library
    img = Image.open(image_path)

    # Create a vector of zeros that will hold our feature vector (the 'avgpool' layer has an output size of 2048)
    feature_vector = torch.zeros(2048)
    
    # Define a function that will copy the output of a layer
    def copy_data(o):
        # Reshape the output tensor to match the shape of the feature_vector
        feature_vector.copy_(o.data.view(feature_vector.shape))
    
    # Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)

    # Run the model on our transformed image
    model(img)

    # Detach our copy function from the layer
    h.remove()
    
    # Return the feature vector
    return feature_vector

def r_similarity(tensor1, tensor2):
    """
    Calculates the reciprocal of the Euclidean distance between two tensors.

    Args:
      tensor1 (torch.Tensor): First input tensor.
      tensor2 (torch.Tensor): Second input tensor.

    Returns:
      float: Reciprocal of the Euclidean distance between two tensors.
    """

    euclidian_distance = float((torch.norm(tensor1 - tensor2)).item())

    # Avoid division by zero
    if euclidian_distance == 0:
      return 1
    
    return 1 / euclidian_distance

def create_ranked_lists(features, total_features):
  """
    Create ranked lists using the feature similarities.

    Args:
      features (list of torch.Tensor): List of feature vectors.
      total_features (int): Total number of features.
    
    Returns:
      2D list: Ranked lists of image indices.
  """

  T = []
  for target_idx in range(total_features):
      
      tq = []
      for query_idx in range(total_features):
          similarity = (query_idx, r_similarity(features[target_idx], features[query_idx]))
          tq.append(similarity)
      
      # Add tq to T sorted by similarity in descending order
      T.append(sorted(tq, key=lambda x: x[1], reverse=True))

  return T

def create_hypergraph(ranked_lists, k):
  """
    Creates a hypergraph based on ranked lists.

    Args:
      ranked_lists (2D list): Ranked lists of image indices.
      k (int): Number of k-nearest neighbors for each target image.
    
    Returns:
      nx.DiGraph: Hypergraph data structure.
  """

  hypergraph = nx.DiGraph() 
  
  for target_idx in range(len(ranked_lists)):
    # Add nodes for each image
    hypergraph.add_node(target_idx)

    for query_idx in range(k):
      # Create hyperedges for the k-nearest neighbors
      hypergraph.add_edge(target_idx, ranked_lists[target_idx][query_idx][0], idx=query_idx+1)

  return hypergraph
def create_incidence_matrix(hypergraph, k):
  """
  Creates an incidence matrix based on a hypergraph.

  Args:
    hypergraph (nx.DiGraph): Hypergraph representation.
    k (int): Number of k-nearest neighbors.
  
  Returns:
    2D list: Incidence matrix.
  """
  
  total_nodes = len(hypergraph.nodes)
  H = np.zeros((total_nodes, total_nodes))

  for node_idx in hypergraph.nodes:
    for neighbor_idx, edge_data in hypergraph[node_idx].items():
      H[node_idx][neighbor_idx] = 1 - math.log(edge_data["idx"], k+1)

  return H

def pairwise_similarity_rel(eq, vi, vj, H):
  """ 
  Compute the weight assigned to the hyperedge eq and the pairwise similarity relationship (p(eq,vi,vj))
  
  Args:
    eq (int): indicate the hyper edge
    vi (int): Hyper node
    vj (int): Hyper node
    H (np array): incident matrix
  """
  
  w_eq = 0
  for i in H[eq]:
    w_eq += i
  return w_eq * H[eq][vi] * H[eq][vj]

def pairwise_cartesian_prod(hypergraph, H):
  """
  Computes the similarity measure based on a hypergraph and an incidence matrix.

  Args:
    hypergraph (hypergraph): The hypergraph with the edges and nodes.
    H (np array): Incident matrix.
  
  Returns:
    2D List: the similarity measure.
  """

  total_nodes = len(hypergraph.nodes)
  C1 = np.zeros((total_nodes, total_nodes))

  for eq in hypergraph.nodes:
    neighbors = [i for i, _ in hypergraph[eq].items()]

    for i in neighbors:
      for j in neighbors:
        C1[i][j] += pairwise_similarity_rel(eq, i, j, H)
        
  return C1
if __name__ == "__main__":

  k = 5
  
  target_images = get_target_image_indices()
  print(f"Target images: {target_images}")
