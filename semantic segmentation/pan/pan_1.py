import torch
import segmentation_models_pytorch as smp

# Returns a Linkenet model which is basically just torch.nn.Module
pan = smp.pan.model.PAN(encoder_name="resnet34",
                       encoder_weights="imagenet",
                       activation="sigmoid",
                       in_channels=3)

# preprocessing input
preprocess = smp.encoders.get_preprocessing_fn('resnet34', pretrained='imagenet')
params = sum(p.numel() for p in pan.parameters())
print("Parameters:", params)
#---------------------------------------------------

#Converting torch to onnx

# Onnx input 

x = torch.randn(2,3, 128, 128)
torch_out = pan(x)
torch.onnx.export(pan,
                 x,
                 "pan.onnx",
                 input_names=["input"],
                 output_names=["output"],
                 opset_version=11)


from openvino.inference_engine import IECore
ie = IECore()

# These files including pan.bin, pan.mapping, pan.xml are
# create after converting the onnx model to openvino through the above step
openvino_pan = ie.read_network(model="pan.xml", weights="pan.bin")
exec_pan = ie.load_network(network=openvino_pan, device_name="CPU", num_requests=1)
openvino_out = exec_pan.infer(inputs={"input": x})

print('\x1b[6;30;42m' + 'Torch output:' + '\x1b[0m', torch_out.detach().numpy())
print('\x1b[6;30;42m' + 'Openvino output:' + '\x1b[0m', openvino_out["output"])
