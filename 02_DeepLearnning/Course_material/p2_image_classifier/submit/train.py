import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import json
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('data_dir', action='store')

parser.add_argument('--save_dir', action='store',
                    dest='save_dir',
                    default='./checkpoint.pth',
                    help='Set checkpoint dir')

parser.add_argument('--arch', action='store',
                    dest='arch',
                    default='vgg13',
                    help='Choose architecture')

parser.add_argument('--learning_rate', action='store',
                    dest='learning_rate',
                    default=0.01,
                    type=float,
                    help='Set learning rate')

parser.add_argument('--hidden_units', action='store',
                    dest='hidden_units',
                    default=4096,
                    type=int,
                    help='Set hidden units number')

parser.add_argument('--epochs', action='store',
                    dest='epochs',
                    default=5,
                    type=int,
                    help='Set epochs')

parser.add_argument('--gpu', action='store_true',
                    default = False,
                    dest='gpu',
                    help='Use GPU for training')

results = parser.parse_args()

def get_args():
    return results

args = get_args()
print(args)

train_dir = args.data_dir + '/train'
valid_dir = args.data_dir + '/valid'
test_dir = args.data_dir + '/test'

train_transforms = transforms.Compose([transforms.Resize(256),
                                transforms.RandomHorizontalFlip(0.3),
                                transforms.RandomRotation(30),
                                transforms.RandomResizedCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([.485, .456, .406],
                                                     [.229, .224, .225])
                               ])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([.485, .456, .406],
                                                     [.229, .224, .225])
                               ])

test_transforms =  transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([.485, .456, .406],
                                                     [.229, .224, .225])
                               ])

# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
validation_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)


train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)
validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=32)

class_to_idx = train_data.class_to_idx
categories = train_data.classes
num_categories = len(categories)

# Build the pre-trained model

if args.arch == 'vgg13':
    model = models.vgg13(pretrained=True)
elif args.arch == 'vgg19':
    model = models.vgg19(pretrained=True)

# Froze the grad
for param in model.parameters():
    param.requires_grad = False

# build the classifier
classifier = nn.Sequential(nn.Linear(25088, args.hidden_units),
                           nn.ReLU(),
                           nn.Dropout(p=0.5),
                           nn.Linear(args.hidden_units, num_categories),
                           nn.LogSoftmax(dim=1))
model.classifier = classifier


def validation(model, validation_loader, criterion):
    if args.gpu:
        model.to('cuda')
    valid_loss = 0
    accuracy = 0
    for data in validation_loader:
        images, labels = data
        if args.gpu:
            images, labels = images.to('cuda'), labels.to('cuda')
        output = model.forward(images)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return valid_loss, accuracy


criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

print_every = 30
steps = 0

if args.gpu:
    model.to('cuda')

for e in range(args.epochs):
    model.train()
    running_loss = 0
    for ii, (inputs, labels) in enumerate(train_loader):
        steps += 1

        if args.gpu:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')

        optimizer.zero_grad()

        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            model.eval()


            with torch.no_grad():
                valid_loss, accuracy = validation(model, validation_loader, criterion)

            print("Epoch: {}/{}.. ".format(e + 1, args.epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss / print_every),
                  "Valid Loss: {:.3f}.. ".format(valid_loss / len(validation_loader)),
                  "Valid Accuracy: {:.3f}".format(accuracy / len(validation_loader)))

            running_loss = 0


            model.train()

model.eval()

with torch.no_grad():
    _, accuracy = validation(model, test_loader, criterion)

print("Test Accuracy: {:.2f}%".format(accuracy * 100 / len(test_loader)))

model.class_to_idx = train_data.class_to_idx

torch.save({
    'class_to_idx': model.class_to_idx,
    'arch': args.arch,
    'state_dict': model.state_dict(),
    'hidden_units': args.hidden_units,
    'optim_state': optimizer.state_dict()
}, args.save_dir)

