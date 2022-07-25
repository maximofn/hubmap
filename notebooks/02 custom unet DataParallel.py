import torch
import torch.nn.functional as F
from pathlib import Path
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import randint
# import tifffile as tiff
import cv2
from tqdm import tqdm

number_of_classes = 2 # 2 classes
number_of_channels = 3
division = 300
resize_img = 800
BS_train = 10
BS_val = 16
EPOCHS = 3
LR = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"


# UNet
def conv3x3_bn(ci, co):
    return torch.nn.Sequential(
        torch.nn.Conv2d(ci, co, 3, padding=1),
        torch.nn.BatchNorm2d(co),
        torch.nn.ReLU(inplace=True)
    )

def encoder_conv(ci, co):
  return torch.nn.Sequential(
        torch.nn.MaxPool2d(2),
        conv3x3_bn(ci, co),
        conv3x3_bn(co, co),
    )

class deconv(torch.nn.Module):
    def __init__(self, ci, co):
        super(deconv, self).__init__()
        self.upsample = torch.nn.ConvTranspose2d(ci, co, 2, stride=2)
        self.conv1 = conv3x3_bn(ci, co)
        self.conv2 = conv3x3_bn(co, co)
    
    # recibe la salida de la capa anetrior y la salida de la etapa
    # correspondiente del encoder
    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX, 0, diffY, 0))
        # concatenamos los tensores
        x = torch.cat([x2, x1], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class UNet(torch.nn.Module):
    def __init__(self, n_classes=number_of_classes, in_ch=number_of_channels, log=False):
        super().__init__()

        self.log = log

        # lista de capas en encoder-decoder con número de filtros
        c = [16, 32, 64, 128]

        # primera capa conv que recibe la imagen
        self.conv1 = torch.nn.Sequential(
          conv3x3_bn(in_ch, c[0]),
          conv3x3_bn(c[0], c[0]),
        )
        # capas del encoder
        self.conv2 = encoder_conv(c[0], c[1])
        self.conv3 = encoder_conv(c[1], c[2])
        self.conv4 = encoder_conv(c[2], c[3])

        # capas del decoder
        self.deconv1 = deconv(c[3],c[2])
        self.deconv2 = deconv(c[2],c[1])
        self.deconv3 = deconv(c[1],c[0])

        # útlima capa conv que nos da la máscara
        self.out = torch.nn.Conv2d(c[0], n_classes, 3, padding=1)

    def forward(self, x):
        if self.log:
            print('\t\tinput shape: ', x.shape)
        # encoder
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x = self.conv4(x3)
        # decoder
        x = self.deconv1(x, x3)
        x = self.deconv2(x, x2)
        x = self.deconv3(x, x1)
        x = self.out(x)
        return x


# Dice coefficient
def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(1)
    m1 = pred.view(num, -1).float()  # Flatten
    m2 = target.view(num, -1).float()  # Flatten
    intersection = (m1 * m2).sum().float()
    dice = (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)
    dice = dice.item()

    return dice


# rle to mask
def rle2mask(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [
        np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])
    ]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo : hi] = 1
    if len(shape) == 3:
        img = img.reshape(shape[0], shape[1])
    else:
        img = img.reshape(shape[0], shape[1])
    return img.T


# Dataset
class Dataset(torch.utils.data.Dataset):
  def __init__(self, dataframe, n_classes=2, dim=2000, interpolation=cv2.INTER_LANCZOS4):
    self.dataframe = dataframe
    self.n_classes = n_classes
    self.dim = dim
    self.interpolation = interpolation

  def __len__(self):
    return len(self.dataframe)

  def __getitem__(self, ix):
    # Get image path from column 'path' in dataframe
    img_path = str(self.dataframe.iloc[ix]['path'])
    # Load image
    img_cv = cv2.imread(img_path)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    # Resize image
    img_cv_res = cv2.resize(img_cv, dsize=(self.dim, self.dim), interpolation=self.interpolation)
    # Normalize image
    img_cv_res_norm = img_cv_res / 255.0
    # Convert to tensor
    img_tensor = torch.from_numpy(img_cv_res_norm).float().permute(2, 0, 1)

    # Get mask
    rle = self.dataframe.iloc[ix]['rle']
    mask_cv = rle2mask(rle, img_cv.shape)
    # Resize mask
    mask_cv_res = cv2.resize(mask_cv, dsize=(self.dim, self.dim), interpolation=self.interpolation)
    # One-hot encode mask
    mask_oh = np.eye(2)[mask_cv_res.astype(int)].astype(np.float32)
    # Convert to tensor
    mask_tensor = torch.from_numpy(mask_oh).float().permute(2, 0, 1)
    
    return img_tensor, mask_tensor


# Fit function
def fit(model, dataloader, epochs=100, lr=3e-4, parallel=False):
    len_int_epochs = len(str(epochs))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    if torch.cuda.device_count() >= 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        if parallel:
            model = torch.nn.DataParallel(model)
        model.cuda()
    else:
        print("Let's use CPU!")
        model.to(device)
    hist = {'loss': [], 'dice': [], 'test_loss': [], 'test_dice': []}
    for epoch in range(epochs):
        bar = tqdm(dataloader['train'])
        train_loss, train_dice = [], []
        model.train()
        for imgs, masks in bar:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            pred_mask = model(imgs)
            loss = criterion(pred_mask, masks)
            loss.backward()
            optimizer.step()
            dice = dice_coeff(pred_mask, masks)
            train_loss.append(loss.item())
            train_dice.append(dice)
            bar.set_description(f"loss {np.mean(train_loss):.5f}, dice {np.mean(train_dice):.5f}")
        hist['loss'].append(np.mean(train_loss))
        hist['dice'].append(np.mean(train_dice))
        bar = tqdm(dataloader['test'])
        test_loss, test_dice = [], []
        model.eval()
        with torch.no_grad():
            for imgs, masks in bar:
                imgs, masks = imgs.to(device), masks.to(device)
                pred_mask = model(imgs)
                loss = criterion(pred_mask, masks)
                dice = dice_coeff(pred_mask, masks)
                test_loss.append(loss.item())
                test_dice.append(dice)
                bar.set_description(f"test_loss {np.mean(test_loss):.5f} test_dice {np.mean(test_dice):.5f}")
        hist['test_loss'].append(np.mean(test_loss))
        hist['test_dice'].append(np.mean(test_dice))
        if len_int_epochs == 1:
            print(f"Epoch {(epoch+1):01d}/{epochs:01d} loss {np.mean(train_loss):.5f} dice {np.mean(train_dice):.5f} test_loss {np.mean(test_loss):.5f} test_dice {np.mean(test_dice):.5f}")
        elif len_int_epochs == 2:
            print(f"Epoch {(epoch+1):02d}/{epochs:02d} loss {np.mean(train_loss):.5f} dice {np.mean(train_dice):.5f} test_loss {np.mean(test_loss):.5f} test_dice {np.mean(test_dice):.5f}")
        elif len_int_epochs == 3:
            print(f"Epoch {(epoch+1):03d}/{epochs:03d} loss {np.mean(train_loss):.5f} dice {np.mean(train_dice):.5f} test_loss {np.mean(test_loss):.5f} test_dice {np.mean(test_dice):.5f}")
        elif len_int_epochs == 4:
            print(f"Epoch {(epoch+1):04d}/{epochs:04d} loss {np.mean(train_loss):.5f} dice {np.mean(train_dice):.5f} test_loss {np.mean(test_loss):.5f} test_dice {np.mean(test_dice):.5f}")
        print("\n\n")
    return hist



def main():
    print(os.getcwd())
    path = Path('./')
    data_path = path / 'data'
    notebooks_path = path / 'notebooks'

    print("Change directory")
    os.chdir(notebooks_path)
    print(os.getcwd())
    path = Path('../')
    data_path = path / 'data'
    notebooks_path = path / 'notebooks'

    print("Load data")
    train_images_path = data_path / "train_images"
    train_df = pd.read_csv(data_path / "train.csv")
    train_df['path'] = train_df.id.apply(lambda x: f'{str(train_images_path)}/{x}.tiff')
    # return

    print("Create dataset")
    dataset = {
        'train': Dataset(train_df[:division], n_classes=2, dim=resize_img),
        'val': Dataset(train_df[division:], n_classes=2, dim=resize_img),
    }
    print(f"Había {len(train_df)} imágenes en el dataset, lo hemos dividido en {len(dataset['train'])} imágenes de entrenamiento y {len(dataset['val'])} imágenes de validación")

    print("Create dataloader")
    dataloader = {
        'train': torch.utils.data.DataLoader(dataset['train'], batch_size=BS_train, shuffle=True, pin_memory=True),
        'test': torch.utils.data.DataLoader(dataset['val'], batch_size=BS_val, pin_memory=True)
    }

    print("Create and fit the model")
    model = UNet()
    hist = fit(model, dataloader, epochs=EPOCHS, lr=LR, parallel=True)
    hist_df = pd.DataFrame(hist)

    print("Test the model")
    sample_val_img, sample_val_mask = next(iter(dataloader['test']))
    one_sample_img = sample_val_img[0]
    one_sample_mask = sample_val_mask[0]
    model.eval()
    with torch.no_grad():
        output = model(one_sample_img.unsqueeze(0).to(device))[0]
        pred_mask = torch.argmax(output, axis=0)
    
    print("Plot train loss and val loss of hist_df")
    plt.figure(figsize=(10, 5))
    plt.plot(hist_df['loss'], label='train_loss')
    plt.plot(hist_df['test_loss'], label='test_loss')
    plt.legend()
    plt.title('Loss')
    plt.show()

    print("Plot train dice and val dice of hist_df")
    plt.figure(figsize=(10, 5))
    plt.plot(hist_df['dice'], label='train_dice')
    plt.plot(hist_df['test_dice'], label='test_dice')
    plt.legend()
    plt.title('Dice')
    plt.show()
        
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30,10))
    ax1.imshow(one_sample_img.permute(1, 2, 0).cpu().numpy())
    ax1.set_title('Image')
    ax2.imshow(torch.argmax(one_sample_mask, axis=0).cpu().numpy())
    ax2.set_title('mask')
    ax3.imshow(pred_mask.squeeze().cpu().numpy())
    ax3.set_title('pred_mask')
    plt.show()
    print(f"output.shape = {output.shape}, pred_mask.shape = {pred_mask.shape}")

main()