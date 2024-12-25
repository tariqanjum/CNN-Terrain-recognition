from flask import Flask, request, render_template
import os
import argparse
import logging
import os
import cv2

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import matplotlib.pyplot as plt
import glob

from unet import UNet
from utils.utils import plot_img_and_mask
from utils.data_loading import BasicDataset
import tensorflow as tf
miou_obj = tf.keras.metrics.MeanIoU(num_classes=6)

TEMPLATE_DIR = os.path.abspath('./templates')
STATIC_DIR = os.path.abspath('./static')

app = Flask(__name__)

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()

def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)

@app.route('/')
def main():
    return render_template("login.html")

@app.route('/login', methods = ['POST'])
def login():
  if request.method == 'POST':
    un = request.form.get('email')
    psd = request.form.get('pass')
    if(un=="admin@gmail.com" and psd=="admin"):
      return render_template("index.html")
    else:
      return render_template("login.html")
  else:
    return render_template("login.html")

@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST':
      f = request.files['file']
      # fname = 'static/'+f.filename
      fname = "static/"+f.filename
      f.save(fname)
      logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
      classes = 7
      bilinear = False
      model = "./checkpoint_epoch1.pth"
      scale = 0.5
      mask_threshold = 0.5
      viz = True
      no_save = True

      net = UNet(n_channels=3, n_classes=classes, bilinear=bilinear)

      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      logging.info(f'Loading model {model}')
      logging.info(f'Using device {device}')

      net.to(device=device)
      state_dict = torch.load(model, map_location=device)
      mask_values = state_dict.pop('mask_values', [0, 1])
      net.load_state_dict(state_dict)

      logging.info('Model loaded!')

      filename = fname
      f_name_msk = "test_images/msk/"+f.filename.split("/")[-1].split(".")[0]+"_mask.png"
      print(filename)
      logging.info(f'Predicting image {filename} ...')
      img = Image.open(filename)
      msk_orig = Image.open(f_name_msk)


      mask = predict_img(net=net,
                          full_img=img,
                          scale_factor=scale,
                          out_threshold=mask_threshold,
                          device=device)

      result = mask_to_image(mask, mask_values)

      fig, ax = plt.subplots(1, 3, figsize=(15, 5))
      ax[0].imshow(img)
      ax[0].axis('off')
      ax[1].imshow(result)
      ax[1].axis('off')
      ax[2].imshow(msk_orig)
      ax[2].axis('off')


      miou_obj.update_state(result, msk_orig)
      print("MIou-->" + str(miou_obj.result().numpy()))

      # result = result().numpy()*40

      numpy_array = np.array(result)
      f_out = np.zeros((numpy_array.shape[0],numpy_array.shape[1],3),dtype='uint8')
      for m in range(0,numpy_array.shape[0]):
         for n in range(0,numpy_array.shape[1]):
            if(numpy_array[m,n]==0):
               f_out[m,n,:] = [17, 141, 215]
            elif(numpy_array[m,n]==1):
               f_out[m,n,:] = [225, 227, 155]
            elif(numpy_array[m,n]==2):
               f_out[m,n,:] = [127, 173, 123]
            elif(numpy_array[m,n]==3):
               f_out[m,n,:] = [185, 122, 87]
            elif(numpy_array[m,n]==4):
               f_out[m,n,:] = [230, 200, 181]
            elif(numpy_array[m,n]==5):
               f_out[m,n,:] = [150, 150, 150]
            elif(numpy_array[m,n]==6):
               f_out[m,n,:] = [193, 190, 175]

               


      numpy_array = numpy_array * 40
      print(type(numpy_array[0,0]))
      print(numpy_array.shape)

      # result.save('static/output.jpg') 

      cv2.imwrite('static/output.jpg',cv2.cvtColor(f_out, cv2.COLOR_RGB2BGR))
    
      return render_template("index.html", input_img = fname, a_del = 'static/output.jpg')

if __name__ == '__main__':
    app.run()