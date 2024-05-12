import torch
import config
device='cuda' if torch.cuda.is_available() else 'cpu'
def compute_recall_precison_f1(num_classes, all_labels, all_preds):
    """
    Compute recall, precision and f1 score for each class and return the macro average of the same.
    """
    classes_present = num_classes
    total_recall = 0
    for i in range(num_classes):
        tp = torch.sum((all_labels == i) & (all_preds == i))
        fn = torch.sum((all_labels == i) & (all_preds != i))
        if tp + fn != 0:
            recall = tp / (tp + fn)
        else:
            recall = 0
        total_recall += recall
    macro_recall = total_recall / classes_present
    print("Macro recall : {:.4f}".format(macro_recall), end=', ')

    # Macro Precision
    total_precision = 0
    for i in range(num_classes):
        tp = torch.sum((all_labels == i) & (all_preds == i))
        fp = torch.sum((all_labels != i) & (all_preds == i))
        if tp + fp != 0:
            precision = tp / (tp + fp)
        else:
            precision = 0
        total_precision += precision
    macro_precision = total_precision / classes_present
    print("Macro precision : {:.4f}".format(macro_precision), end=', ')
    macro_f1 = (2 * macro_precision * macro_recall) / (macro_precision + macro_recall)
    print("Macro f1 : {:.4f}".format(macro_f1))
    return macro_recall, macro_precision, macro_f1

@torch.no_grad()
def calculate_metrics(eval_dataloader,seq_model):
    """
    Calculate the accuracy, precision, recall and f1 score for the model on the input dataloader
    """
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    seq_model.eval()
    correct = 0
    total = 0
    total_loss=0
    # Initialize the lists to store the labels and predictions for all points in the dataloader
    all_labels = torch.zeros(len(eval_dataloader.dataset))
    all_preds = torch.zeros(len(eval_dataloader.dataset))
    for i,(images, labels) in enumerate(eval_dataloader):
                # Move the batch to the GPU if available
                curr_index = i*config.batch_size
                images = images.to(device)
                seq_model.to(device)
                # Forward pass and prediction
                outputs = seq_model(images)
                outputs = outputs.cpu().detach()
                _, predicted_label = torch.max(outputs.data, 1)
                labels= labels.cpu().detach()
                #computing loss
                total_loss+=cross_entropy_loss(outputs,labels)

                # Compute accuracy
                total += labels.size(0)
                correct += (predicted_label == labels).sum().item()
                # Store the labels and predictions for current batch
                all_labels[curr_index:(curr_index+len(labels))] = labels
                all_preds[curr_index:(curr_index+len(labels))] = predicted_label
            #average loss
    average_loss=(total_loss/len(eval_dataloader))
    print('Avergae loss {:.4f}'.format(average_loss),end = ' , ')
    # Validation Accuracy
    val_accuracy = 100 * correct / total
    print('Accuracy: {:.2f}%'.format(val_accuracy),end = ' , ')
    #macro recall
    macro_recall, macro_precision,macro_f1=compute_recall_precison_f1(config.num_classes,all_labels,all_preds)
    out = {
                'accuracy': val_accuracy,
                'precision': macro_precision,
                'recall': macro_recall,
                'f1_score': macro_f1,
                'loss': average_loss
            }
    seq_model.train()
    return out
