#### Import libraries
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import argparse


## set working directory and define data directories

## list of arguments
parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', help = 'input the name of your dataset')
parser.add_argument('--category_dict', help = 'input your label dictionary json file (include format)')

#### list of models to use in this program
model_dict = {'vgg16': models.vgg16(pretrained = True),
             'vgg13': models.vgg13(pretrained = True),
             'densenet121': models.densenet121(pretrained = True)}

parser.add_argument('--arch', choices = model_dict.keys(), help = 'input your neural network architecture model from the dictionary displayed')

## model hyperparameters

parser.add_argument('--output_units', help = 'input number of output units', type = int)

parser.add_argument('--learning_rate', help = 'input learning rate', type = float)

parser.add_argument('--epochs', help = 'input number of output units', type = int)

parser.add_argument('--save_dir', help = 'name of checkpoint directory')

args = parser.parse_args()

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

## Define transforms for the training, validation, and testing sets

## means and standard deviations for the normalizations
means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]

## training and validation: apply random scaling (224), cropping, and flipping
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.CenterCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(means, stds)])

## testing: apply resize and crop to 224
test_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(means, stds)])

val_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(means, stds)])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform= train_transforms)
test_data = datasets.ImageFolder(test_dir, transform= test_transforms)
val_data = datasets.ImageFolder(valid_dir, transform = val_transforms)

# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size= 64, shuffle = True)
testloader = torch.utils.data.DataLoader(test_data, batch_size= 64)
valloader = torch.utils.data.DataLoader(val_data, batch_size= 64)

#### label mapping
import json


with open(args.category_dict, 'r') as f:
    cat_to_name = json.load(f)
    
#### Building and Training

## define model
model = model_dict[args.arch]
print(model)
### print model in order to edit the sequential classifier accordingly

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
## freeze paramerters
for param in model.parameters():
    param.requires_grad = True
    
##### build classifier
## insert input_units, hidden_units, and output_units
## also, add more layers depending on the architecture 

if args.arch == 'densenet121':
    input_units = model.classifier.in_features
    hidden_units = 500
    classifier = nn.Sequential(nn.Linear(input_units, hidden_units),
                                nn.ReLU(),
                                nn.Dropout(p = 0.3),
                               nn.Linear(hidden_units,args.output_units),
                               nn.LogSoftmax(dim = 1))
else:
    input_units = model.classifier[0].in_features
    hidden_units = model.classifier[0].out_features
    classifier = nn.Sequential(nn.Linear(input_units, hidden_units),
                                nn.ReLU(),
                                nn.Dropout(p = 0.3),
                               nn.Linear(hidden_units,hidden_units),
                                nn.ReLU(),
                                nn.Dropout(p = 0.3),
                               nn.Linear(hidden_units,args.output_units),
                               nn.LogSoftmax(dim = 1))
    
    
model.classifier = classifier
print(model.classifier)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = args.learning_rate)

model.to(device);


##### Train model with validation
epochs = args.epochs
steps = 0
running_loss = 0
print_every = 5

for e in range(epochs):
    for images, labels in trainloader:
        steps += 1
        
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        log_ps = model.forward(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        
        if steps % print_every == 0:
            val_loss = 0 ## validation loss
            accuracy = 0
         
            model.eval()

            with torch.no_grad():
                for images, labels in valloader:
                    images, labels = images.to(device), labels.to(device)

                    log_ps = model.forward(images)
                    loss = criterion(log_ps, labels)

                    val_loss += loss.item()

                    ## calculate accuracy
                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim = 1)
                    equals = (top_class == labels.view(*top_class.shape))

                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                    
            print(f"Epoch: {e+1}/{epochs}.."
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {val_loss/len(valloader):.3f}.. "
                  f"Accuracy: {accuracy/len(testloader):.3f}")

            running_loss = 0
            model.train() 
        
#### train with test 
epochs = 1
steps = 0
running_loss = 0
print_every = 5

for e in range(epochs):
    for images, labels in trainloader:
        steps += 1
        
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        log_ps = model.forward(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0 ## test loss
            accuracy = 0
            
            model.eval()

            with torch.no_grad():
                for images, labels in testloader:
                    images, labels = images.to(device), labels.to(device)

                    log_ps = model.forward(images)
                    loss = criterion(log_ps, labels)

                    test_loss += loss.item()

                    ## calculate accuracy
                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim = 1)
                    equals = (top_class == labels.view(*top_class.shape))

                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    #print(test_loss)
 
            print(f"Epoch: {e+1}/{epochs}..",
                  f"Train loss: {running_loss/print_every:.3f}.. ",
                  f"Test loss: {test_loss/len(testloader):.3f}.. ",
                  f"Accuracy: {accuracy/len(testloader):.3f}")
            running_loss = 0
            model.train() 
        
# Save the checkpoint 
model.class_to_idx = train_data.class_to_idx

def save_checkpoint(file_path):
    checkpoint = {'epochs': args.epochs,
                  'arch': args.arch,
                  'input_size': input_units,
                  'hidden_units': hidden_units,
                  'output_units': args.output_units,
                  'learning_rate': args.learning_rate,
                  'model': model,
                  'classifier': model.classifier,
                 'state_dict': model.state_dict(),
                 'class_to_idx': model.class_to_idx,
                 'optimizer': optimizer.state_dict
                 }
    torch.save(checkpoint, file_path)
    print('The model is saved in', file_path)

save_checkpoint(args.save_dir)