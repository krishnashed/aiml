import io
import os
import hashlib
import tempfile
import requests
import numpy as np
from PIL import Image
from argparse import ArgumentParser

import torch
import torchvision.transforms as transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# for example we are using the Resnet18
from torchvision import models
#------------------------------------------------------------------------------------------------------------------------------------------------
# define helper functions
def fetch(url):
  # efficient loading of URLS
  fp = os.path.join(tempfile.gettempdir(), hashlib.md5(url.encode('utf-8')).hexdigest())
  if os.path.isfile(fp) and os.stat(fp).st_size > 0:
    with open(fp, "rb") as f:
      dat = f.read()
  else:
    print("fetching", url)
    dat = requests.get(url).content
    with open(fp+".tmp", "wb") as f:
      f.write(dat)
    os.rename(fp+".tmp", fp)
  return dat

def get_image_net_labels():
  with open("labels.txt", 'r') as f:
    labels_map = [x.split(sep=' ', maxsplit=1)[-1].strip()[10:] for x in f]
  return labels_map
#--------------------------------------------------------------------------------------------------------------------------------------------
labels = get_image_net_labels()
#--------------------------------------------------------------------------------------------------------------------------------------------
#resnet18 = models.resnet18(pretrained=True)
#resnet34 = models.resnet34(pretrained=True)
resnet50 = models.resnet50(pretrained=True)
#resnet101 = models.resnet101(pretrained=True)
#resnet152 = models.resnet152(pretrained=True)
#vgg16 = models.vgg16(pretrained=True)
#vgg19 = models.vgg16(pretrained=True)
#squeezenet = models.squeezenet1_0(pretrained=True)
#mobilenet = models.mobilenet_v2(pretrained=True)
#inceptionv3 = models.inception_v3(pretrained=True)
#-------------------------------------------------------------------------------------------------------------------------------------------
model = models.resnet50(pretrained=True)       #depending on the model change models.<name>
_ = model.eval()
#-------------------------------------------------------------------------------------------------------------------------------------------
# define the image transformations to apply to each image
image_transformations = transforms.Compose([
    transforms.Resize(256),                               # resize to a (256,256) image
    transforms.CenterCrop(224),                           # crop centre part of image to get (244, 244) grid
    transforms.ToTensor(),                                # convert to tensor
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),    # normalise image according to imagenet valuess
])
#-----------------------------------------------------------------------------------------------------------------------------------------
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/07._Camel_Profile%2C_near_Silverton%2C_NSW%2C_07.07.2007.jpg/1200px-07._Camel_Profile%2C_near_Silverton%2C_NSW%2C_07.07.2007.jpg"#"some/url/to/some/image.jpg"
image = Image.open(io.BytesIO(fetch(url)))  # load any image
x = image_transformations(image) # [3, H, W]
x = x.view(1, *x.size())
print(x.size())
##image.show()    #written to see the image

#------------------------------------------------------------------------------------------------------------------------------------------
with torch.no_grad():
    out = model(x)
#----------------------------------------------------------------------------------------------------------------------------------------
# here we are getting the label of input image as we have used url of camel we get highest score of camel.
# keep the top 10 scores
number_top = 10
probs, indices = torch.topk(out, number_top)
print(f"Top {number_top} results:")
print("===============")
for p, i in zip(probs[0], indices[0]):
    print(p, "--", labels[i])
#------------------------------------------------------------------------------------------------------------------------------------------
print("-"*70)
print("Now we are converting this pytorch model to ONNX model")
torch.onnx._export(model, x, f'resnet50.onnx', export_params=True)   #convert model name as <name>.onnx
print(f"See in the folder we have a resnet50.onnx file")
print("-"*70)
#------------------------------------------------------------------------------------------------------------------------------------------
from openvino.inference_engine import IECore
#------------------------------------------------------------------------------------------------------------------------------------------
#model_xml = "fp32/resnet18.xml"    #model xml file (can be opted using run {mo --input_mode <model_name>.onnx }  in command promt)
#model_bin = os.path.splitext(model_xml)[0] + ".bin"
model_xml="resnet50.xml"
model_bin = "resnet50.bin"
#model_xml, model_bin

#print(model_xml)
#print("#--------------------------------------------------------------------------------#")
#print(model_bin)
#-------------------------------------------------------------------------------------------------------------------------------------------
# Plugin initialization for specified device and load extensions library if specified.
print("Creating Inference Engine...")
ie = IECore()

# Read IR
print("Loading network")
#net = ie.read_network(args.model, os.path.splitext(args.model)[0] + ".bin")

#lets load model and weights of model manually 
net = ie.read_network(model_xml, model_bin)

print("Loading IR to the plugin...")
exec_net = ie.load_network(network=net, device_name="CPU", num_requests=2)
print(f"exec_net: {exec_net}")
print("-"*70)
#-------------------------------------------------------------------------------------------------------------------------------------------
# convert to numpy to pass this to the IECore
x = x.numpy()
#--------------------------------------------------------------------------------------------------------------------------------------------
# this is a bit tricky. So the input to the model is the input from ONNX graph
# IECore makes a networkX graph of the "computation graph" and when we run .infer
# it passes it through. If you are unsure of what to pass you can always check the
# <model>.xml file. In case of pytorch models the value "input.1" is the usual
# suspect. Happy Hunting!
out = exec_net.infer(inputs={"input.1": x})

# the output looks like this {"node_id": array()} so we simply load the output
out = list(out.values())[0]
print("Output Shape:", out.shape)
print("-"*70)
#--------------------------------------------------------------------------------------------------------------------------------------------
# keep the top 10 scores
out = out[0]
number_top = 10
indices = np.argsort(out, -1)[::-1][:number_top]
probs = out[indices]
print(f"Top {number_top} results:")
print("===============")
for p, i in zip(probs, indices):
    print(p, "--", labels[i])
    print("-"*70)
#----------------------------------------------------------------------------------------------------------------------------------------------
