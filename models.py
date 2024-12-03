import torch
import torch.nn as nn


class VGG11Net(nn.Module):
    def __init__(self, num_classes):
        super(VGG11Net, self).__init__()
        # Declare all the layers for feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            # nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.BatchNorm2d(512),
            # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2),


        )
        
        # Declare all the layers for classification
        self.classifier = nn.Sequential(
            nn.Linear(32768, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),

            # nn.Linear(128, 64),
            # nn.ReLU(),
            # nn.BatchNorm1d(64),
            # nn.Dropout(0.3),
            # nn.Linear(64, 32),
            # nn.ReLU(),
            # nn.BatchNorm1d(32),
            # nn.Dropout(0.3),
            # nn.Linear(32, 16),
            # nn.ReLU(),
            # nn.BatchNorm1d(16),
            # nn.Dropout(0.3),
            # nn.Linear(4096, 2048),
            # nn.ReLU(),
            # nn.BatchNorm1d(2048),
            # nn.Linear(2048, 1024),
            # nn.ReLU(),
            # nn.BatchNorm1d(1024),
            # nn.Linear(1024, 512),
            # nn.ReLU(),
            # nn.BatchNorm1d(512),
            # nn.Dropout(0.3),
            # nn.Linear(512, 256),
            # nn.ReLU(),
            # nn.BatchNorm1d(256),
            # nn.Dropout(0.3),
            # nn.Linear(256, 128),
            # nn.ReLU(),
            # nn.BatchNorm1d(128),
            # nn.Dropout(0.3),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1),
            #nn.Softmax(dim=1)
        )

    def forward(self, x):
        # Apply the feature extractor in the input
        x = self.features(x)
        
        # Squeeze the three spatial dimentions in one
        x = torch.flatten(x, 1)
        
        # Classifiy the image
        x = self.classifier(x)
        return x
    


class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        # Declare all the layers for feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),

            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),

            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),

        
        )
        
        # Declare all the layers for classification
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # Apply the feature extractor in the input
        x = self.features(x)
        
        # Squeeze the three spatial dimentions in one
        x = torch.flatten(x, 1)
        
        # Classifiy the image
        x = self.classifier(x)
        return x
    
