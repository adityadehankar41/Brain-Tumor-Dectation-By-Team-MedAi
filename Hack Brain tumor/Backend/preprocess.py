from torchvision import transforms
import torch
from PIL import Image
import numpy as np
import nibabel as nib
import cv2


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])


def preprocess_image(pil_img):
    return transform(pil_img).unsqueeze(0)

def preprocess_nifti_to_slices(nifti_path, axis=2, step=4):
    """Load NIfTI, extract slices along axis, return list of PIL images (grayscale->RGB)"""
    img = nib.load(nifti_path).get_fdata()
    img = np.nan_to_num(img)
    # normalize 0-255 per volume
    mn, mx = img.min(), img.max()  
    if mx - mn > 0:
        img = (img - mn) / (mx - mn) * 255.0
    img = img.astype('uint8')
    slices = []
    for i in range(0, img.shape[axis], step):
        if axis==0:
            sl = img[i,:,:]
        elif axis==1:
            sl = img[:,i,:]
        else:
            sl = img[:,:,i]
        # convert to 3-channel
        sl = cv2.resize(sl, (224,224))
        sl = cv2.equalizeHist(sl)
        pil = Image.fromarray(sl).convert('RGB')
        slices.append(pil)
    return slices
