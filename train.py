import random
import numpy as np
import torch
import torchvision.transforms.v2 as T2
import torchvision.transforms as T
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler
from models import VGG11Net, AlexNet

import intel_extension_for_pytorch as ipex


device = 'xpu' if torch.xpu.is_available() else 'cpu' # Utilise la carte graphique, si GPU intel est disponible


NB_CLASSES = 8 
CLASSES = ['battery', 'cardboard', 'glass', 'metal', 'organic', 'paper', 'plastic', 'textile']
NB_EPOCHS = 100
BATCH_SIZE = 64
DTYPE = torch.float32
LR = 0.001
L1_REG = 0.01
DATA = "./waste-classification-challenge/train/train/"

train_transform = T2.Compose(
    [

        T2.ToImage(),
        T2.ToDtype(DTYPE),
        T2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T2.RandomHorizontalFlip(),
        T2.RandomVerticalFlip(),
        T2.RandomRotation(degrees=30),
        #T2.RandomAffine(degrees=0),
        T2.RandomChannelPermutation(),
        T2.RandomResizedCrop(256, scale=(0.6, 1), antialias=True),
        T2.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5), saturation=(0.5, 1.5)),
        T2.GaussianBlur(kernel_size=(3, 9)),
 
    ]
).to(device)

test_transform = T2.Compose(
    [
        T2.ToImage(),
        T2.ToDtype(DTYPE),
        T2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T2.Resize(256, antialias=True),
        T2.CenterCrop(256),
    ]
).to(device)

dataset = ImageFolder(root=DATA)


model = VGG11Net(num_classes=NB_CLASSES)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=0.01)
model = model.to(device)
criterion = criterion.to(device)
if device == 'xpu':
    model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=DTYPE)

def train_epoch(model, train_loader):
    ########## Training ##########
    model.train()

    train_n_corrects = 0
    train_n_total = 0
    train_losses = []

    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            loss = criterion(output, target)
            _, predicted = torch.max(output, 1)

            # L1_term = torch.tensor(0., requires_grad=True)
            # for name, weights in model.named_parameters():
            #     if 'bias' not in name:
            #         weights_sum = torch.sum(torch.abs(weights))
            #     L1_term = L1_term + weights_sum

            # L1_term = L1_term / nweights
            # loss = loss + L1_REG * L1_term


        train_n_corrects += (predicted == target).sum().item()
        train_n_total += target.numel()
        train_losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_epoch_acc = train_n_corrects / train_n_total
    train_epoch_loss = np.mean(train_losses).__float__()

    ########## Print results ##########
    print(f"Train accuracy: {train_epoch_acc}")
    print(f"Train loss: {train_epoch_loss}")
    print("--------------------")

    return model, train_epoch_acc, train_epoch_loss

def test_model(model, test_loader, val_values, val_predictions):
    ########## Testing ##########
        model.eval()
        test_n_corrects = 0
        test_n_total = 0
        test_losses = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                loss = criterion(output, target)
                _, predicted = torch.max(output, 1)

                val_values.append(target)
                val_predictions.append(predicted)
                test_n_corrects += (predicted == target).sum().item()
                test_n_total += target.numel()
                test_losses.append(loss.item())

        test_epoch_acc = test_n_corrects / test_n_total
        test_epoch_loss = np.mean(test_losses).__float__()

        #### Print results ####
        print(f"Test accuracy: {test_n_corrects / test_n_total}")
        print(f"Test loss: {np.mean(test_losses)}")
        print("--------------------")

        return test_epoch_acc, test_epoch_loss, val_values, val_predictions




def train_model(full_dataset):

    if full_dataset:
        dataset.transform = train_transform
        sampler = RandomSampler(dataset)
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)

    elif not full_dataset:
        train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])
        train_dataset.dataset.transform = train_transform
        test_dataset.dataset.transform = test_transform
        train_sampler = RandomSampler(train_dataset)
        test_sampler = RandomSampler(test_dataset)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, sampler=test_sampler)


    train_graph_acc = []
    train_graph_loss = []
    test_graph_acc = []
    test_graph_loss = []

    val_values = []
    val_predictions = []


    # nweights = 0
    # for name,weights in model.named_parameters():
    #     if 'bias' not in name:
    #         nweights = nweights + weights.numel()
        

    for epoch in range(NB_EPOCHS):
        print(f"Epoch {epoch}")
        model, train_epoch_acc, train_epoch_loss = train_epoch(model, train_loader)

        if not full_dataset:
            test_epoch_acc, test_epoch_loss = test_model(model, test_loader)
            
            train_graph_acc.append(train_epoch_acc)
            train_graph_loss.append(train_epoch_loss)
            test_graph_acc.append(test_epoch_acc)
            test_graph_loss.append(test_epoch_loss)

            ########## Visualise results ##########
            x = np.arange(1, NB_EPOCHS + 1)
            fig, ax1 = plt.subplots()
            ax1.set_xlabel('Number of epochs')
            ax1.set_ylabel('Accuracy')
            ax1.plot(x, train_graph_acc, label='Train Accuracy', color='tab:blue')
            ax1.plot(x, test_graph_acc, label='Test Accuracy', color='tab:cyan')
            ax1.tick_params(axis='y')
            ax2 = ax1.twinx()
            ax2.set_ylabel('Loss')
            ax2.plot(x, train_graph_loss, label='Train Loss', color='tab:green')
            ax2.plot(x, test_graph_loss, label='Test Loss', color='tab:olive')
            ax2.tick_params(axis='y')
            fig.legend()
            plt.savefig('graphs/performance.png')

            plt.figure()
            cm = confusion_matrix(torch.cat(val_values).cpu(), torch.cat(val_predictions).cpu())
            cm_display = ConfusionMatrixDisplay(cm, display_labels=CLASSES)
            cm_display.plot(xticks_rotation='vertical')
            plt.savefig('graphs/confusion_matrix.png')

            

train_model(full_dataset=False)

########## Save the model ##########
torch.save(model.state_dict(), 'model.pth')



    
    



