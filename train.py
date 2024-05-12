import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import v2
import torch.utils.data as data
from torchvision.datasets import ImageFolder
from models import lenet, resnet, efficient_net;
import warnings
warnings.filterwarnings("ignore")
import warnings
from datetime import datetime
import config
import random
import argparse
import wandb
wandb_log = True

# Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from tqdm import tqdm
from performance_measure import compute_recall_precison_f1,calculate_metrics
import os
torch.manual_seed(42);
torch.cuda.manual_seed(42);


data_transforms = v2.Compose([
        v2.Grayscale(),
        v2.RandomRotation(15, expand=True),
        v2.ColorJitter(brightness=(1,15)),
        v2.GaussianBlur(5),
        v2.Resize((config.reshape_height, config.reshape_width)),
        v2.ToTensor(),
        #v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        v2.Normalize(mean=[0.5], std=[0.5])
    ])

# Loading train ,test and validation data
train_path = os.path.join('project_files',config.synthetic_folder_name, 'synthetic_train')
val_path = os.path.join('project_files',config.synthetic_folder_name, 'synthetic_val')
num_classes = config.num_classes
train_dataset = ImageFolder(train_path, transform=data_transforms)
train_loader = data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_dataset = ImageFolder(val_path, transform=data_transforms)
val_loader = data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
print('Number of points in train dataset - ', len(train_dataset))
print('Number of points in validation dataset - ', len(val_dataset))
####
test_path=  os.path.join('project_files',config.synthetic_folder_name, 'test')
test_dataset = ImageFolder(test_path, transform=data_transforms)
test_loader = data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
print('Number of points in test dataset - ', len(test_dataset))
####
eval_path = os.path.join('sample_eval_data')
eval_dataset = ImageFolder(eval_path, transform=data_transforms)
eval_loader = data.DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False)
print('Number of files in eval dataset ',len(eval_dataset))
####

def create_train_subset_loader(eval_batches):

    train_dataset = ImageFolder(train_path, transform=data_transforms)
    #generating 30 random batches for inference of training set
    random_indices = random.sample(range(len(train_dataset)), eval_batches * config.batch_size)
    subset_dataset = torch.utils.data.Subset(train_dataset, random_indices)
    subset_loader = torch.utils.data.DataLoader(subset_dataset, batch_size=config.batch_size, shuffle=True)
    return subset_loader


def model_train(n_epochs,train_loader,val_loader,seq_model,criterion,optimizer,best_val_f1=-10000,device='cpu'):
    
    for epoch in range(n_epochs):
        # Set the model to train mode

        # Loop over the training data in batches
        for i, (images, labels) in enumerate(train_loader):
            # Move the batch to the GPU if available
            images = images.to(device)  # batch_size,1,resize_height,resize_width
            #print(images.shape)
            labels = labels.to(device) # 1xbatch_size - [9,3,4,......]
            seq_model.to(device)
            # Forward pass
            outputs = seq_model(images)   # [batch_size, num_classes]
            loss = criterion(outputs, labels)

            # Backward pass and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            
            if(i+1)%100==0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, n_epochs, i+1, len(train_loader), loss.item()))
        train_subset_loader=create_train_subset_loader(config.eval_batches)
        print('Train',end=' - ')
        out_train=calculate_metrics(train_subset_loader,seq_model)
        print('Val',end=' - ')
        out_val=calculate_metrics(val_loader,seq_model)
        print('Test',end= ' - ')
        out_test=calculate_metrics(test_loader,seq_model)
        print('eval',end= ' - ')
        out_eval=calculate_metrics(eval_loader,seq_model)
        if wandb_log:
            try:
                wandb.log(
                    {
                        "train_accuracy": out_train["accuracy"],
                        "train_f1_score": out_train["f1_score"],
                        "train_loss": out_train["loss"],
                        "val_accuracy": out_val["accuracy"],
                        "val_f1_score": out_val["f1_score"],
                        "val_loss": out_val["loss"],
                    }
                )
            except Exception as e:
                print(f"logging to wandb failed: {e}")
       #saving the best model
        if(out_val["f1_score"]>best_val_f1):
           best_val_f1=out_val["f1_score"]
           print(f'Saving model with val f1 score {best_val_f1}')
           checkpoint = {
                    "model": seq_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_f1": best_val_f1,
                    "config": config,
                }
           torch.save(seq_model.state_dict(), os.path.join(out_dir,'state.pt'))
           torch.save(optimizer.state_dict(), os.path.join(out_dir,'optim.pt'))
           torch.save(best_val_f1,os.path.join(out_dir,'best_val_f1.pt'))
           torch.save(data_transforms,os.path.join(out_dir,'transforms.pt'))
                       

parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--model', type=str, choices=['lenet', 'resnet','enet'], default='lenet',
                    help='Choose the model architecture (lenet , resnet or efficient net)')
parser.add_argument('--resume', type=str, choices=['yes', 'no'], default='no',
                    help='Specify whether to resume the training or train from scratch')
args = parser.parse_args()

if args.model == 'lenet':
    seq_model = lenet.LeNet(config.num_classes)
    out_dir='lenet_saved_model'
    num_params = sum(p.numel() for p in seq_model.parameters() if p.requires_grad)
    print(f'Number of parameters in the LeNet model: {num_params/1e6} M')
elif args.model == 'resnet':
        resnet_model = resnet.CustomResNet(config.num_classes)
        seq_model = resnet_model.get_model()
        out_dir='resnet_saved_model'
elif args.model == 'enet':
        enet_model = efficient_net.CustomEfficientNet(config.num_classes)
        seq_model = enet_model.get_model()
        out_dir='enet_saved_model'

else:
    raise ValueError('Invalid model architecture. Choose between "lenet" and "resnet".')
config_dict = {key: value for key, value in vars(config).items() if not key.startswith('__')}
os.makedirs(out_dir, exist_ok=True)
if args.resume == 'no':
    print('Training model from scratch.')
    seq_model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(seq_model.parameters(), lr=config.learning_rate)
    n_epochs = config.epochs

    print('Training Started with following configuration')
    print(f'Epochs: {config.epochs}, Learning rate: {config.learning_rate}, '
        f'Batch size: {config.batch_size}, Reshape height: {config.reshape_height}, '
        f'Reshape width: {config.reshape_width}, Model: {args.model}')
    #logging data to wandb
    
    
    
    wandb_project = "font-classifier-project"
    wandb_run_name = out_dir + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if wandb_log:
        wandb.init(project=wandb_project, name=wandb_run_name, config=config_dict)

    model_train(config.epochs, train_loader, val_loader, seq_model, criterion, optimizer)
elif args.resume == 'yes':
    print('Loading pre-trained model and continuing training.')
    wandb_project = "font-classifier-project"
    wandb_run_name = out_dir + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    wandb_run_name = f'{out_dir}_resume' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if wandb_log:
        wandb.init(project=wandb_project, name=wandb_run_name, config=config_dict)
    seq_model.load_state_dict(torch.load(os.path.join(out_dir,'state.pt')))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(seq_model.parameters(), lr=config.learning_rate)
    optimizer.load_state_dict(torch.load(os.path.join(out_dir,'optim.pt')))
    best_val_f1 = torch.load(os.path.join(out_dir,'best_val_f1.pt'))
    best_val_f1=(best_val_f1.item())
    print('previous best f1 score ',best_val_f1)
    seq_model.to(device)
    model_train(config.epochs, train_loader, val_loader, seq_model, criterion, optimizer,best_val_f1)
    