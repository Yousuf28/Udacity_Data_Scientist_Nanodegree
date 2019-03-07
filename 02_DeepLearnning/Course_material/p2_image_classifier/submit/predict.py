import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import json

import os

import argparse


parser = argparse.ArgumentParser()

parser.add_argument('image_path', action='store')

parser.add_argument('checkpoint', action='store')

parser.add_argument('--top_k ', action='store',
                    dest='top_k',
                    default=5,
                    type=int,
                    help='Return top K most likely classes')

parser.add_argument('--category_names ', action='store',
                    dest='category_names',
                    default='cat_to_name.json',
                    help='Use a mapping of categories to real names')

parser.add_argument('--gpu', action='store_true',
                    default = False,
                    dest='gpu',
                    help='Use GPU for training')

results = parser.parse_args()

def get_args():
    return results

args = get_args()
print(args)




def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['arch'] == 'vgg19':
        model = models.vgg19()
    classifier = nn.Sequential(nn.Linear(25088, checkpoint['hidden_units']),
                               nn.ReLU(),
                               nn.Dropout(p=0.5),
                               nn.Linear(checkpoint['hidden_units'], 102),
                               nn.LogSoftmax(dim=1))
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    optimizer.load_state_dict(checkpoint['optim_state'])

    return model, optimizer


# load model
model, optimizer = load_checkpoint(args.checkpoint)


def process_image(image):

    im = Image.open(image)
    size = 256, 256
    im = im.resize(size)
    box = (16, 16, 240, 240)
    im = im.crop(box)

    np_im = np.array(im)
    np_im = np_im / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_im = (np_im - mean) / std
    np_im = np_im.transpose((2, 0, 1))
    return np_im


def predict(image_path, model, topk=5):

    np_img = process_image(image_path)
    model.eval()
    img = torch.zeros(1, 3, 224, 224)
    img[0] = torch.from_numpy(np_img)
    
    if args.gpu and torch.cuda.is_available():
     print('Gpu Available')
    if args.gpu:
     model.to('cuda')
     img = img.to('cuda')
    with torch.no_grad():
        output = model(img)

    output = output.to('cpu')
    ps = torch.exp(output).topk(topk)
    p, c = ps
    idx_to_class = dict(zip(model.class_to_idx.values(), model.class_to_idx.keys()))
    c = [idx_to_class[i] for i in c[0].numpy()]

    return p[0].numpy(), c

# make prediction
probs, classes = predict(args.image_path, model, topk=args.top_k)

# load category file
names = classes
if args.category_names:
    if os.path.isfile(args.category_names):
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        names = np.array([cat_to_name[i] for i in classes])

# print the result
for result in zip(names, probs):
    name, prob = result
    print(name, prob)