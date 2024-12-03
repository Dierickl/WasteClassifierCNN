import os
import pandas as pd
import torch
import torchvision.transforms.v2 as T2
from models import VGG11Net
from PIL import Image
from torch.utils.data import Dataset, DataLoader


import intel_extension_for_pytorch as ipex


device = 'xpu' if torch.xpu.is_available() else 'cpu' # Utilise la carte graphique, si GPU intel est disponible


LR = 0.001
DATA = "./waste-classification-challenge/test/test/"
DTYPE = torch.float32
BATCH_SIZE = 64
CLASSES = ['battery', 'cardboard', 'glass', 'metal', 'organic', 'paper', 'plastic', 'textile']

class WasteSubmissionDataset(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.image_list = os.listdir(main_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.image_list[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        img_id = os.path.splitext(os.path.basename(img_loc))[0]
        return tensor_image, img_id

transform = T2.Compose(
    [
        T2.ToImage(),
        T2.ToDtype(DTYPE),
        T2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T2.Resize(256, antialias=True),
        T2.CenterCrop(256),
        
    ]
).to(device)

dataset = WasteSubmissionDataset(main_dir=DATA, transform=transform)
submission_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = VGG11Net(num_classes=8)
model.load_state_dict(torch.load("model.pth", map_location=device))

model = model.to(device, dtype=DTYPE)
# if device == 'xpu':
#     model = ipex.optimize(model, dtype=DTYPE)
model.eval()

images_id = []
predictions = []

with torch.no_grad():
    for batch_idx, (img, tuple_img_id) in enumerate(submission_loader):
        img = img.to(device)
        output = model(img)
        prob = torch.nn.functional.softmax(output, dim=1)
        _, predicted = torch.max(output, 1)
        for i, img_id  in enumerate(tuple_img_id):
            images_id.append(img_id)
            predictions.append(CLASSES[predicted[i]])


df = pd.DataFrame({'ID': images_id, 'Label': predictions})
df.to_csv('submission.csv', index=False)
    




    
    



