import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
import torchvision

from PIL import Image
import requests
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device} is found")

class VGG(nn.Module):
  def __init__(self):
    super(VGG, self).__init__()
    self.layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2', # content representation
            '28': 'conv5_1',
            }
    self.model = model = models.vgg19(pretrained=True).features

  def forward(self, x):
    features = {}
    for layer_num, layer in enumerate(self.model):
      x = layer(x)
      if str(layer_num) in list(self.layers.keys()):
        features[self.layers[str(layer_num)]] = x

    return features

def load_image(img_path, max_size=400, shape=None):
  if "http" in img_path:
      response = requests.get(img_path)
      image = Image.open(BytesIO(response.content)).convert('RGB')
  else:
      image = Image.open(img_path).convert('RGB')
  
  if max(image.size) > max_size:
      size = max_size
  else:
      size = max(image.size)
  
  if shape is not None:
      size = shape
      
  in_transform = transforms.Compose([
                      transforms.Resize(size),
                      transforms.ToTensor()
                      ])

  # discard the transparent, alpha channel (that's the :3) and add the batch dimension
  image = in_transform(image)[:3,:,:].unsqueeze(0)
  image = image.to(device)
  
  return image

def gram_matrix(tensor):
  b, d, h, w = tensor.size()
  tensor = tensor.view(d, h*w)
  gram = torch.mm(tensor, tensor.t())
  return gram

content = load_image('https://pytorch.org/tutorials/_static/img/neural-style/dancing.jpg')
style = load_image('https://pytorch.org/tutorials/_static/img/neural-style/picasso.jpg', shape=content.shape[-2:])

total_steps = 3000
learning_rate = 0.003
style_weights = {'conv1_1': 1.,
                 'conv2_1': 0.8,
                 'conv3_1': 0.5,
                 'conv4_1': 0.3,
                 'conv5_1': 0.1}
alpha = 1
beta = 1e6

model = VGG().to(device).eval()

generated = content.clone().requires_grad_(True)
optimizer = optim.Adam([generated], lr=learning_rate)


for step in range(1, total_steps+1):
  content_features = model(content)
  style_features = model(style)
  style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
  generated_features = model(generated)
  content_loss = torch.mean((generated_features['conv4_2'] - content_features['conv4_2'])**2)

  style_loss = 0
  for layer in style_weights:
    generated_feature = generated_features[layer]
    _, d, h, w = generated_feature.shape
    generated_gram = gram_matrix(generated_feature)
    style_gram = style_grams[layer]
    layer_style_loss = style_weights[layer] * torch.mean((generated_gram - style_gram)**2)

    style_loss += layer_style_loss / (2*d*h*w)
  
  total_loss = alpha*content_loss + beta*style_loss

  optimizer.zero_grad()
  total_loss.backward()
  optimizer.step()

  if step % 100 == 0:
    save_image(generated, 'generated.jpg') 
    print(f"Output is saved for step: {step}")





  