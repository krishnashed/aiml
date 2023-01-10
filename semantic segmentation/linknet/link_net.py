import torch
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt 
import segmentation_models_pytorch as smp
from torchvision import transforms
#import tensorflow as tf

# Returns a Linkenet model which is basically just torch.nn.Module
Linknet = smp.Linknet(encoder_name="resnet34",
                       encoder_weights="imagenet",
                       activation="sigmoid",
                       in_channels=3)

# preprocessing input
preprocess = smp.encoders.get_preprocessing_fn('resnet34', pretrained='imagenet')
params = sum(p.numel() for p in Linknet.parameters())
print("Parameters:", params)
#---------------------------------------

img = Image.open("test.png")
temp_img = img
plt.imshow(img)
plt.title("Model Input Image")
plt.show()

# Image transformations for model input 
m, s = np.mean(img, axis=(0, 1)), np.std(img, axis=(0,1))
transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize(mean=m ,std=s)])

# image resizing for model input 
inp_tensor = transform(img)
#inp_tensor =torch.reshape(inp_tensor, (1,3, 256,256))   #reshape to increase a dimensio
inp_batch = inp_tensor.unsqueeze(0)
#print(tf.shape(inp_batch))
#A copy to use with onnx later
onnx_inp = inp_batch.clone()
openvino_inp = inp_batch.numpy()
#--------------------------------------------------

with torch.no_grad():
    out = Linknet(inp_batch)
#---------------------------------------------------------------------------------------------------------------------------------
# printing model output 
out = torch.squeeze(out, 0)
out = torch.squeeze(out, 0)
out_numpy = out.detach().cpu().numpy()
plt.imshow(out_numpy)
plt.title("Linknet Model output Image")
plt.show()
torch.onnx.export(Linknet,
                 onnx_inp,
                 "Linknet.onnx",
                 input_names=["input"],
                 output_names=["output"])

from openvino.inference_engine import IECore
ie = IECore()

# These files including Linknet.bin, Linknet.mapping, Linknet.xml are
# create after converting the onnx model to openvino through the above step
openvino_linknet = ie.read_network(model="Linknet.xml", weights="Linknet.bin")
exec_linknet = ie.load_network(network=openvino_linknet, device_name="CPU", num_requests=1)
openvino_out = exec_linknet.infer(inputs={"input": openvino_inp})
openvino_out = np.squeeze(openvino_out["output"], axis=(0,1))
plt.imshow(openvino_out)
plt.title("Link net openvino Model output Image")
plt.show()
