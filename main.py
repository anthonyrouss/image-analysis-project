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

def find_score_of_image(i, v, T):
  """
  Finds the score of image index 'v' in the hyperedge 'T[i]'.

  Args:
    i (int): Index of the target object.
    v (int): Index of the query object.
    T (2D list): Ranked lists.
  
  Return:
    int: Score of 'v' in the ranked list, or -1 if not found.
  """

  score = -1
  for j in range(len(T[i])):
    if (T[i][j][0] == v):
      score = T[i][j][1]
      break
  return score

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

def rank_normalization(L, T):
  """"
  Creates a symmetric similarity matrix based on ranked lists.

  Args:
    L (int): Length of the ranked lists.
    T (2D list): Ranked lists.

  Returns:
    2D list: Normalized ranked lists.
  """

  normalized_T = []

  for i in range(L):
    temp = []

    for j in range(L):
      score = 2 * L - (find_score_of_image(i,j,T) + find_score_of_image(j,i,T))
      temp.append((j, score))
    normalized_T.append(temp)

  # Sort each sublist by similarity score
  normalized_T = [sorted(t,key = lambda x: x[1]) for t in normalized_T]

  return normalized_T

def hypergraph_manifold_ranking(features, max_iters=15, k=3):
  """
  Perform hypergraph manifold ranking.

  Args:
    features (list): List of features extracted from each image.
    max_iters (int): Number of maximum iterations.
    k (int): Neighborhood size.
  
  Returns:
    2D list: Normalized ranked lists.
  """

  L = len(features)
  # Initialize ranked lists
  T = create_ranked_lists(features, L)

  for _ in range(max_iters):

    # Perform Rank Normalization
    T = rank_normalization(L, T)

    # Create the hypergraph
    hypergraph = create_hypergraph(T,k)

    # Create the incidence matrix 'H'
    H = np.array(create_incidence_matrix(hypergraph, k))

    # Calculate similarity matrices
    Sn = H @ H.T
    Sv = H.T @ H
    
    # Calculate Hadamard product (element-wise multiplication)
    S = np.multiply(Sn, Sv)

    # Compute pairwise cartesian product
    C_prod = pairwise_cartesian_prod(hypergraph, H)

    # Compute the affinity matrix 'W'
    W = np.multiply(C_prod, S)

    # Update T based on affinity matrix
    for i in range(len(features)):
      for j in range(len(features)):
        T[i][j] = (T[i][j][0], W[i][T[i][j][0]])

    # Sort each sublist
    T = [sorted(t,key = lambda x: x[1], reverse=True) for t in T]

  return T

def calculate_accuracy(ranked_lists, labels, k, target_images):
  """"
  Calculates the accuracy of the ranked lists for a set of target images.

  Args:
    ranked_lists (2D list): Ranked lists of image indices.
    labels (list): Labels of the images.
    k (int): Number of nearest neighbors.
    target_images (list): Indices of target images.

  Returns:
    list: List of accuracy scores corresponding to each target image.    
  """

  # Sum of integers from 1 to k
  sum = k * (k+1) // 2
  
  accuracy_scores = []
  for i in target_images:
    m = 0
    accuracy = 0
    for j in ranked_lists[i][:k]:
      if labels[j[0]] == labels[i]:
        accuracy += (k-m)/sum
      else:
        accuracy += 0
      m += 1
    accuracy_scores.append(accuracy)
    
  return accuracy_scores

def display(ranked_list, accuracy):
  """
  Display images along with information in a matplotlib subplot.

  Parameters:
  - ranked_list (list): A list of tuples containing image indices and scores.
  - accuracy (float): The accuracy value.
  """

  _, axs = plt.subplots(1, len(ranked_list), figsize=(14, 4))
  plt.gcf().canvas.manager.set_window_title(f"Target image {ranked_list[0][0]}")

  for ax, (img_idx, score) in zip(axs, ranked_list):
      image = np.transpose(C[img_idx], (1, 2, 0))
      ax.imshow(image)
      title = f"Label: {labels[img_idx]}\nScore: {score:.3f}"
      if img_idx == ranked_list[0][0]:
        title += "\n(target image)"

      ax.set_title(title)
      ax.axis('off')

  plt.suptitle(f"Accuracy: {accuracy:.3f}")
  plt.show()

if __name__ == "__main__":

  k = 5
  
  target_images = get_target_image_indices()
  print(f"Target images: {target_images}")

  final_ranked_lists = hypergraph_manifold_ranking(C, k=k)
  accuracy_list = calculate_accuracy(final_ranked_lists, labels, k, target_images)

  target_ranked_lists = [final_ranked_lists[i][:k] for i in target_images]
  for i in range(len(target_ranked_lists)):
    ranked_list = target_ranked_lists[i]
    display(ranked_list, accuracy_list[i])

  for i in range(len(target_images)):
    print(f"t{target_images[i]} = {final_ranked_lists[target_images[i]][0:k]}, accuracy={accuracy_list[i]:.3f}")