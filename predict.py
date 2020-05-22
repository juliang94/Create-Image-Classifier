## import libraries
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import argparse

## set argumants
parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', help = 'input the name of your dataset')

parser.add_argument('--save_dir', help = 'input the name of yout checkpoint file')

parser.add_argument('--category_dict', help = 'input your label dictionary json file (include format)')

parser.add_argument('--image_dir', help = 'input the image directory from the test folder')

args = parser.parse_args()

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

### load checkpoint
def load_checkpoint(file_path):
    
    checkpoint = torch.load(file_path)
    
    model = checkpoint['model']
    arch = checkpoint['arch']
    input_size = checkpoint['input_size']
    output_size = checkpoint['output_units']
    lr = checkpoint['learning_rate']
    epochs = checkpoint['epochs']
    
    
    for param in model.parameters():
        param.requires_grad = False  
    
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.optimizer = checkpoint['optimizer'] 
    
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

model2 = load_checkpoint(args.save_dir)
print(model2)

## get labels
import json

with open(args.category_dict, 'r') as f:
    cat_to_name = json.load(f)

## image processing
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch mode
    img = Image.open(image)
    ## resize image
    im_thn = img.resize((256, 256))
    
    width_rs, height_rs = im_thn.size ## width and height of resize
    
    ## crop out the center 224x224 of the image
    width_cr, height_cr = (224, 224) ## width and height of cropped image
    
    left = (width_rs - width_cr) / 2
    right = (left + width_cr)
    
    top = (height_rs - height_cr) / 2 
    bottom = (top + height_cr)
    
    ## crop image
    im_cr = im_thn.crop((left, top, right, bottom))
    
    ## get colors
    np_image = np.array(im_cr)/255 ## convert to floats between 0 and 1
    
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    
    norm_img = (np_image - means)/stds ## normalize images
    
    img_reorder = norm_img.transpose((2, 0, 1))
    img_final = torch.from_numpy(img_reorder) ## convert to tensor
    return img_final

## image show function
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

img = process_image(test_dir + '/1/image_06754.jpg')
imshow(img)

## predict probabilities of image 
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    ## input image
    flower_in = process_image(image_path)
    
    ## convert to cuda
    images = flower_in.type(torch.cuda.FloatTensor) 
    
    ## unsqueeze input images
    images = images.unsqueeze(0)
    
    ## run model with image
    model.eval()
    with torch.no_grad():

        log_ps = model.forward(images)
        ps = torch.exp(log_ps)
        top_p, top_classes = ps.topk(topk, dim = 1)
        
        ## convert tensors to lists
        top_p = top_p.tolist()[0] 
        top_classes = top_classes.tolist()[0]
        
        ## list of labels turned to strings
        class_list = list(map(str, top_classes))
        
        ## classes converted to flower names
        top_labels = [cat_to_name[x] for x in class_list]
        

    return top_p, top_classes, top_labels
    
probs, classes, labels = predict((test_dir + args.image_dir), model= model2)
print(probs)
print(classes)
print(labels)

## Do the Sanity Chack
#Display an image along with the top 5 classes
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

import torch.nn.functional as F

def sanity_check(img_path, model):
    

    
    img = process_image(img_path)
    show_img = imshow(img)
    
    fig, ax = plt.subplots()
    
    probs, classes, labels = predict(img_path, model, topk = 5)
    
    ax.barh(labels, probs)
    ax.set_yticklabels(labels)
    ax.set_ylabel('Flower Species')
    ax.set_xlabel('Probability')
    ax.set_title('Probability of Flower Names')
    plt.show()
    
sanity_check((test_dir + args.image_dir), model= model2)