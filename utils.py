import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision.models import ResNet18_Weights
from tqdm import tqdm
from torchvision import models

class Results:
    def __init__(self, time_elapsed, best_acc, model, optimizer, criterion, num_epochs, batch_size, dataset_sizes):
        self.time_elapsed = time_elapsed
        self.best_acc = best_acc
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.dataset_sizes = dataset_sizes

def save_confusion_matrix(cm, class_names, filename='confusion_matrix.png'):
    # Define the figure
    plt.figure(figsize=(10, 10))

    # Define the confusion matrix
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)

    # Define the title
    plt.title('Confusion matrix')

    # Define the labels
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Save the figure
    plt.savefig(filename)


# def save_confusion_matrix(cm, class_names, filename='confusion_matrix.png'):
#     # Define the figure
#     plt.figure(figsize=(10, 10))

#     # Define the confusion matrix
#     plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

#     # Define the title
#     plt.title('Confusion matrix')

#     # Define the color bar
#     plt.colorbar()

#     # Define the ticks
#     tick_marks = np.arange(len(class_names))
#     plt.xticks(tick_marks, class_names, rotation=45)
#     plt.yticks(tick_marks, class_names)

#     # Define the text
#     thresh = cm.max() / 2.
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             plt.text(j, i, int(cm[i, j]), horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')

#     # Define the labels
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')

#     # Save the figure
#     plt.savefig(filename)

# plot train_loss and val_loss, train_acc and val_acc in the same graph by epoch and save to image
def save_training_graph(history, filename='training_graph.png'):
    # Define the figure
    plt.figure(figsize=(16, 8))

    # Define the train loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='train loss')
    plt.plot(history['val_loss'], label='val loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')

    # Define the train accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='train acc')
    plt.plot(history['val_acc'], label='val acc')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')

    # Save the figure
    plt.savefig(filename)


def save_results(params: Results, filename='training_results.txt'):
    with open(filename, 'w') as f:
        f.write(f'Training complete in {params.time_elapsed // 60}m {params.time_elapsed % 60}s\n')
        f.write(f'Best val Acc: {params.best_acc}\n')
        f.write(f'Best model weights saved to model.pth\n')
        f.write(f'Model name used: {params.model.__class__.__name__}\n')
        f.write(f'Optimizer Name: {params.optimizer.__class__.__name__}\n')
        f.write(f'Loss function: {params.criterion}\n')
        f.write(f'Epochs: {params.num_epochs}\n')
        f.write(f'Batch size: {params.batch_size}\n')
        f.write(f'Dataset sizes: {params.dataset_sizes}\n')

def load_model_resnet(device, model_path='model.pth'):
    if os.path.exists(model_path):
        bar = tqdm(total=1, desc='Loading trained model', position=0, leave=True)
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        bar.update(1)
        return model
    return None

def load_model_efficientnet(device, model_path='model.pth'):
    from efficientnet_pytorch import EfficientNet
    if os.path.exists(model_path):
        bar = tqdm(total=1, desc='Loading trained model', position=0, leave=True)
        model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=1)
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        bar.update(1)
        return model
    return None

# def exponential_smoothing(data, alpha):
#     smoothed_data = [data[0]]
#     for i in range(1, len(data)):
#         smoothed_data.append(alpha * data[i] + (1 - alpha) * smoothed_data[-1])
#     return smoothed_data