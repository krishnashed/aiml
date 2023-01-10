import torch
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt 
import segmentation_models_pytorch as smp
from torchvision import transforms
#import tensorflow as tf

# Returns a Linkenet model which is basically just torch.nn.Module
fpn = smp.FPN(encoder_name="resnet34",
                       encoder_weights="imagenet",
                       activation="sigmoid",
                       in_channels=3)

# preprocessing input
preprocess = smp.encoders.get_preprocessing_fn('resnet34', pretrained='imagenet')
params = sum(p.numel() for p in fpn.parameters())
print("Parameters:", params)

#----------------------------------------------------
#Loading a plotting input image
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
    out = fpn(inp_batch)
#---------------------------------------------------------------------------------------------------------------------------------
# printing model output 
out = torch.squeeze(out, 0)
out = torch.squeeze(out, 0)
out_numpy = out.detach().cpu().numpy()
plt.imshow(out_numpy)
plt.title("Torch Model output Image")
plt.show()
#--------------------------------
# torch to onnx 


torch.onnx.export(fpn,
                 onnx_inp,
                 "fpn.onnx",
                 input_names=["input"],
                 output_names=["output"],
                 opset_version=11)



#----------------------------------------------------------------------------------------------------------------------------------
from openvino.inference_engine import IECore
ie = IECore()

# These files including fpn.bin, fpn.mapping, fpn.xml are
# create after converting the onnx model to openvino through the above step
openvino_fpn = ie.read_network(model="fpn.xml", weights="fpn.bin")
exec_fpn = ie.load_network(network=openvino_fpn, device_name="CPU", num_requests=1)
openvino_out = exec_fpn.infer(inputs={"input": openvino_inp})

openvino_out = np.squeeze(openvino_out["output"], axis=(0,1))
plt.imshow(openvino_out)
plt.title("openvino Model output Image")
plt.show()


