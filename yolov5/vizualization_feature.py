
import sys
# Add the parent directory of 'yolov5' to the Python path
sys.path.append('C:/Users/anduw/Desktop/my_project/Experiment')

import numpy as np
import torch
from torch.autograd import Variable
import sklearn
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from yolov5.models.yolo import Model

import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import argparse
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import matplotlib as mpl
import urllib.request
import random
import seaborn as sns
from PIL import Image

parser = argparse.ArgumentParser(description='PyTorch t-SNE for STL10')
parser.add_argument('--save-dir', type=str, default='./results', help='path to save the t-sne image')
# parser.add_argument('--batch-size', type=int, default=128, help='batch size (default: 128)')
parser.add_argument('--seed', type=int, default=1, help='random seed value (default: 1)')

args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 1
transform = transforms.Compose([
    transforms.Resize((1280, 1280)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])

source_img = 'C:/Users/anduw/Desktop/my_project/Experiment/yolov5/datax/samples/0/aachen_000011_000019_leftImg8bit.png'
target_img = 'C:/Users/anduw/Desktop/my_project/Experiment/yolov5/datax/samples/1/aachen_000011_000019_leftImg8bit_foggy_beta_0.02.png'

# config_path = '/home/andualemwelabo.tulu/ADV_DA/yolov5/models/yolov5s.yaml'
# weight_path_tdda = '/home/andualemwelabo.tulu/ADV_DA/yolov5/runs/train/Melina_DIT_001/weights/best.pt'
#weight_path_bsln = '/home/andualemwelabo.tulu/ADV_DA/yolov5/runs/train/Meli_BSLN_002/weights/best.pt'

# Model
model = Model("C:/Users/anduw/Desktop/my_project/Experiment/yolov5/models/yolov5s.yaml")
model = torch.load('C:/Users/anduw/Desktop/my_project/Experiment/yolov5/runs/train/kit_001b/weights/best.pt', map_location=torch.device('cpu'))['model']  # local model

model.to(device)
backbone = model.model[0:10]  # Adjust the indexing based on your model's structure
backbone.to(device)

# Loading image
s_image1 = Image.open(source_img)
t_image2 = Image.open(target_img)

image1 = transform(s_image1)
image2 = transform(t_image2)

print(image1.shape)
# adding batch dimension
source = image1.unsqueeze(0) # Add a batch dimension
print(source.shape)
target = image2.unsqueeze(0) # Add a batch dimension

source = Variable(source.to(device).half())
target = Variable(target.to(device).half())

input_tensor = torch.cat([source, target], dim=0)

# Passing the imported image through the feature extractor
backbone.eval()
# Assuming 'images' is a tensor containing your input images
with torch.no_grad():
    features = backbone(input_tensor)

# Separate the feature maps for source and target
features_source = features[0]  # Assuming source is the first image
features_target = features[1]  # Assuming target is the second image

# Print the shapes after passing through the backbone
print("Source feature map shape:", features_source.shape)
print("Target feature map shape:", features_target.shape)

# Flatten the feature maps to [num_samples, num_features]
features_source = features_source.view(512, 40 * 40).detach().cpu().numpy()
features_target = features_target.view(512, 40 * 40).detach().cpu().numpy()

print(features_source.shape)

feature_maps = [
    features_source,  # Example feature map 1
    features_target  # Example feature map 2
]

# Concatenate feature maps for t-SNE
concatenated_feature_maps = np.concatenate(feature_maps, axis=0)
print(concatenated_feature_maps.shape)

# Combine the feature maps and create labels
labels = np.concatenate((np.zeros(features_source.shape[0]), np.ones(features_target.shape[0])))

# Create a DataFrame for visualization
df_subset = pd.DataFrame(concatenated_feature_maps, columns=[f'pixel{i}' for i in range(concatenated_feature_maps.shape[1])])
df_subset['label'] = labels

# Perform t-SNE to reduce the combined feature maps to 2 dimensions
time_start = time.time()
tsne = TSNE(n_components=2, perplexity=40, learning_rate=250, n_iter=1000, n_iter_without_progress=300, random_state=4, init='pca')
tsne_result = tsne.fit_transform(concatenated_feature_maps)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

# Update DataFrame with t-SNE results
df_subset['tsne-2d-one'] = tsne_result[:, 0]
df_subset['tsne-2d-two'] = tsne_result[:, 1]

# Plot the scatterplot
plt.rcParams['figure.figsize'] = 10, 10
sns.scatterplot(
    x='tsne-2d-one', y='tsne-2d-two',
    hue='label',  # Change 'y' to 'label'
    palette=sns.color_palette("tab10"),
    data=df_subset,
    marker='o',
    legend="full",
    alpha=0.9
)

# Adjust x and y axis limits if needed
# plt.xlim(-200, 200)
# plt.ylim(-200, 200)

plt.xticks([])
plt.yticks([])
plt.xlabel('')
plt.ylabel('')

# Save the plot
save_dir = 'C:/Users/anduw/Desktop/my_project/Experiment/yolov5/runs/'
plt.savefig(os.path.join(save_dir, 'BSLN_FEAT_000bs.png'), bbox_inches='tight')
print('done!')

# https://github.com/2-Chae/PyTorch-tSNE/blob/main/main.py
