Before Running the model file:

Install openvino by:  pip install openvino-dev[onnx,pytorch]==2022.1.0

this works
 pip install openvino-dev==2021.4.2

step2
from command promt run   "mo --input_model <INPUT_MODEL>.onnx"  to get .xml   and .bin file



after above two steps run the python code to use the openvino model