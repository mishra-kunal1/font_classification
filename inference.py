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
    print('Number of points in dataset - ', len(dataset))
    return dataloader

def inference_test():
    model_dict={'lenet':'lenet','resnet':'resnet','resnet256':'resnet256',#'efficient_net':'enet'
            }
    for model in model_dict.keys():
        model_name=model_dict[model]
        print('Model - ',model_name)
        out_dir=os.path.join(f'saved_models',f'{model_name}_saved_model')
        data_transforms=torch.load(os.path.join(out_dir,f'transforms_{model_name}.pt'))
        test_loader=get_dataloader(path_test_folder,data_transforms)
        if(model=='lenet'):
            seq_model = lenet.LeNet(config.num_classes)
        elif(model=='resnet' or model=='resnet256'):
            seq_model = resnet.CustomResNet(config.num_classes).get_model()
        elif(model=='efficient_net'):
            enet_model = efficient_net.CustomEfficientNet(config.num_classes).get_model()
        seq_model.load_state_dict(torch.load(os.path.join(out_dir,f'state_{model_name}.pt'),map_location=torch.device('cpu')))
        #calcualte the metrics 5 times and take the average
        out=calculate_metrics(test_loader,seq_model,5,False)
        #take average of the metrics
        print('*'*50)
        print('Accuracy: {:.2f}%'.format(out['accuracy']),end = ' , ')
        print('Precision: {:.2f}'.format(out['precision']),end = ' , ')
        print('Recall: {:.2f}'.format(out['recall']),end = ' , ')
        print('F1 Score: {:.2f}'.format(out['f1_score']),end = ' , ')
        print('Loss: {:.4f}'.format(out['loss']))
        print('*'*50)
   

if __name__ == "__main__":
    #path_test_folder='project_files/synthetic_data_15_3/test'
    path_test_folder=''
    #check if the folder exists or path_test_folder is empty
    #ge total number of files in the folder
    num_files=len(os.listdir(path_test_folder))
    if  path_test_folder=='' or num_files==0:
        print('Please provide a valid path to the test folder')
        exit()
    inference_test()
    
    