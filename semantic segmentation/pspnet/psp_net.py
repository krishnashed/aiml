import torch
import segmentation_models_pytorch as smp

# Returns a Linkenet model which is basically just torch.nn.Module
pspnet = smp.PSPNet(encoder_name="resnet34",
                       encoder_weights="imagenet",
                       activation="sigmoid",
                       in_channels=3)

# preprocessing input
preprocess = smp.encoders.get_preprocessing_fn('resnet34', pretrained='imagenet')

#Cannot be converted from torch to onnx

# Onnx input 

x = torch.randn(1,3, 320, 320)
torch_out = pspnet(x)
torch.onnx.export(pspnet,
                  x,
                  "pspnet.onnx",
                  input_names=["input"],
                  output_names=["output"],
                  opset_version=11)
                  #operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)

from openvino.inference_engine import IECore
ie = IECore()

# These files including pspnet.bin, pspnet.mapping, pspnet.xml are
# create after converting the onnx model to openvino through the above step
openvino_pspnet = ie.read_network(model="pspnet.xml", weights="pspnet.bin")
exec_pspnet = ie.load_network(network=openvino_pspnet, device_name="CPU", num_requests=1)
openvino_out = exec_pspnet.infer(inputs={"input": x})

print("torch_output:", torch_out.detach().numpy())
print("openvino_output:", openvino_out["output"])
