from performance_measure import calculate_metrics
from models import lenet,resnet,efficient_net
import config
import torch
from torchvision.transforms import v2
import torch.utils.data as data
from torchvision.datasets import ImageFolder
import os

def get_dataloader(path,data_transform):
    dataset=ImageFolder(path,transform=data_transform)
    dataloader=data.DataLoader(dataset,batch_size=config.batch_size)
    return dataloader

def inference_test():
    """
    Function to evaluate the models on the test data
    """
    models=['lenet','resnet','resnet256','enet']
    print('Total files in the test folder:',len(os.listdir(test_data_path)))
    for model in models:
        model_name=model
        print('Starting Evaluation with Model - ',model_name)
        out_dir=os.path.join(f'saved_models',f'{model_name}_saved_model')
        data_transforms=torch.load(os.path.join(out_dir,f'transforms_{model_name}.pt'))
        test_loader=get_dataloader(test_data_path,data_transforms)
        if(model=='lenet'):
            seq_model = lenet.LeNet(config.num_classes)
        elif(model=='resnet' or model=='resnet256'):
            seq_model = resnet.CustomResNet(config.num_classes).get_model()
        elif(model=='efficient_net'):
            enet_model = efficient_net.CustomEfficientNet(config.num_classes).get_model()
        if(model=='enet'):
            #having issue with loading efficient net model
            #works fine during training but not during inference
            seq_model.load_state_dict(torch.load(os.path.join(out_dir,f'state_{model_name}.pt'),map_location=torch.device('cpu')),strict=False)
        else:
            seq_model.load_state_dict(torch.load(os.path.join(out_dir,f'state_{model_name}.pt'),map_location=torch.device('cpu')))
        #calcualte the metrics 5 times and take the average
        out=calculate_metrics(test_loader,seq_model,3,True)
        #take average of the metrics
        print('-'*50)
        print('Final Score for ',model_name)
        print('Accuracy: {:.2f}%'.format(out['accuracy']))
        print('Precision: {:.2f}'.format(out['precision']))
        print('Recall: {:.2f}'.format(out['recall']))
        print('F1 Score: {:.2f}'.format(out['f1_score']))
        print('Loss: {:.4f}'.format(out['loss']))
        print('-'*50)
        print('*'*50)
        print('-'*50)
        

if __name__ == "__main__":
    
    #update the path to the test folder
    test_data_path=''
    
    if  test_data_path=='' or not os.path.exists(test_data_path):
        print('Please provide a valid path to the test folder')
        exit()
    inference_test()
    
    