# Importing libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights
import time
import os
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from utils import save_confusion_matrix, save_training_graph, save_results, load_model, save_image_names_after_test, Results
from efficientnet_pytorch import EfficientNet

params = {
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
    'data_dir': 'data',
    'results_dir': 'results',
    'num_classes': 2,
    'model': {
        'name': 'efficientnet-b7', # resnet18 or efficientnet-b0
    },
    'input_size': 224,
}

# Train the model epochs and save the best model. Use the best model to test the model and generate the confusion matrix. Use tqdm to show the progress bar. Create a training and validation graphing the loss and accuracy.
def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, device, batch_size, epochs):
    # Define the start time
    since = time.time()

    # Define the best model weights
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Define the history
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # Iterate over the epochs
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        # Iterate over the training and validation phases
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            # Define the running loss and corrects
            running_loss = 0.0
            running_corrects = 0

            # Use TQDM for progress bar
            dataloader = tqdm(dataloaders[phase], total=len(dataloaders[phase]))

            # Iterate over the data
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward and optimize
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Update TQDM progress bar
                dataloader.set_postfix({'loss': loss.item(), 'acc': torch.sum(preds == labels.data).item() / len(labels)})

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # save the history
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    # Define the end time
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Save the training graph
    save_training_graph(history, filename='results/training_graph.png')

    # save a text with parameters, model used, results and more
    save_results(Results(
        time_elapsed=time_elapsed, 
        best_acc=best_acc, 
        model=model, 
        optimizer=optimizer, 
        criterion=criterion,
        num_epochs=epochs, 
        batch_size=batch_size, 
        dataset_sizes=dataset_sizes,
        device=device,
        input_size=params['input_size']
    ), filename='results/training_results.txt')
    

    return model

# Test the model and generate the confusion matrix using matplotlib and export to image
def test_model(model, dataloaders, dataset_sizes, device, class_names=['pinus', 'not pinus']):
    # Define the model in evaluation
    model.eval()

    # Define the running corrects
    running_corrects = 0

    # Define the confusion matrix
    confusion_matrix = torch.zeros(2, 2)

     # Get image names and labels
    image_names = [sample[0] for sample in dataloaders['test'].dataset.samples]
    labels = dataloaders['test'].dataset.targets

    image_names_correct = []
    image_names_incorrect = []

    # Iterate over the data
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        # Statistics
        running_corrects += torch.sum(preds == labels.data)

        # Get image names
        batch_image_names = dataloaders['test'].dataset.samples

        for t, p in zip(labels.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

    for i in range(len(labels)):
        if labels[i] == preds[i]:
            image_names_correct.append(image_names[i])
        else:
            image_names_incorrect.append(image_names[i])

    # Define the accuracy
    accuracy = running_corrects.double() / dataset_sizes['test']
    print('Test Accuracy: {:.4f}'.format(accuracy))

    # Export the confusion matrix
    save_confusion_matrix(confusion_matrix, class_names, filename='results/confusion_matrix.png')

    # Save the image names
    save_image_names_after_test(image_names_correct, image_names_incorrect)

def main(params, only_test=False):
    # Define the data directory
    data_dir = params['data_dir']

    # Results directory
    os.makedirs(params['results_dir'], exist_ok=True)

    # Define the data transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(params['input_size']),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(params['input_size']),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(params['input_size']),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Define the training, validation and test directories
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val', 'test']}

    # Define the dataloaders
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=params['batch_size'], shuffle=True) for x in ['train', 'val']}
    dataloaders['test'] = torch.utils.data.DataLoader(image_datasets['test'], batch_size=params['batch_size'], shuffle=False)

    # Define the dataset sizes
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

    # Define the class names
    class_names = image_datasets['train'].classes

    # Define the model
    if only_test:
        model = load_model(params, model_path='results/model.pth')
    elif params['model']['name'] == 'resnet18':
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, params['num_classes'])
        model = model.to(params['device'])
    elif params['model']['name'] == 'efficientnet-b0':
        model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=params['num_classes'])
        model = model.to(params['device'])
    elif params['model']['name'] == 'efficientnet-b7':
        model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=params['num_classes'])
        model = model.to(params['device'])

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model, if necessary
    if not only_test:
        model = train_model(model, criterion, optimizer, dataloaders, dataset_sizes, params['device'], params['batch_size'], epochs=params['epochs'])
        torch.save(model.state_dict(), 'results/model.pth')

    # Test the model
    test_model(model, dataloaders, dataset_sizes, params['device'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--test', action='store_true', help='Only test the model')
    args = parser.parse_args()

    params['epochs'] = args.epochs
    params['batch_size'] = args.batch_size

    # Clear the terminal
    os.system('cls' if os.name == 'nt' else 'clear')

    print('Parameters:', params)

    # Main function
    main(params, only_test=args.test)