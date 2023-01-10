import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt 
#%matplotlib inline
#---------------------------------------------------------------------------------------------------------------------------------
unet = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=1, init_features=32, pretrained=True)
params = sum(p.numel() for p in unet.parameters())
print("Parameters:", params)
#---------------------------------------------------------------------------------------------------------------------------------
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
inp_batch = inp_tensor.unsqueeze(0)

#A copy to use with onnx later
onnx_inp = inp_batch.clone()
openvino_inp = inp_batch.numpy()
#---------------------------------------------------------------------------------------------------------------------------------
# Processing with the model
with torch.no_grad():
    out = unet(inp_batch)
#---------------------------------------------------------------------------------------------------------------------------------
# printing model output 
out = torch.squeeze(out, 0)
out = torch.squeeze(out, 0)
out_numpy = out.detach().cpu().numpy()
plt.imshow(out_numpy)
plt.title("Torch Model output Image")
plt.show()
#---------------------------------------------------------------------------------------------------------------------------------
# torch to onnx 
torch.onnx.export(unet,
                  onnx_inp,
                  "unet.onnx",
                  verbose=False,
                  input_names=["input"],
                  output_names=["output"])
                  
#---------------------------------------------------------------------------------------------------------------------------------
from openvino.inference_engine import IECore
#---------------------------------------------------------------------------------------------------------------------------------
ie = IECore()
openvino_unet = ie.read_network(model="unet.xml", weights="unet.bin")
exec_net = ie.load_network(network=openvino_unet, device_name="CPU", num_requests=1)
openvino_out = exec_net.infer(inputs={"input": openvino_inp})
#---------------------------------------------------------------------------------------------------------------------------------
openvino_out = np.squeeze(openvino_out["output"], axis=(0,1))
plt.imshow(openvino_out)
plt.title("openvino Model output Image")
plt.show()
#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------
