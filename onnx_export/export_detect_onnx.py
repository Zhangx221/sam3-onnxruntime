import torch
from transformers import Sam3Processor, Sam3Model
import cv2

device = "cpu"

model = Sam3Model.from_pretrained("models").to(device)
processor = Sam3Processor.from_pretrained("models")

model.eval()

image_url = "./onnx_export/zidane.jpg"
output_file = "models/onnx_detect/sam3.onnx"

os.makedirs(os.path.dirname(output_file), exist_ok=True)

image = cv2.imread(image_url)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

inputs = processor(images=image, text="ear", return_tensors="pt").to(device)

pixel_values = inputs["pixel_values"]
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
print(input_ids.shape)

class Sam3ONNXWrapper(torch.nn.Module):
    def __init__(self, sam3):
        super().__init__()
        self.sam3 = sam3

    def forward(self, pixel_values, input_ids, attention_mask):
        outputs = self.sam3(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.pred_masks, outputs.pred_boxes, outputs.pred_logits

wrapper = Sam3ONNXWrapper(model).to(device).eval()

torch.onnx.export(
    wrapper,
    (pixel_values, input_ids, attention_mask),
    output_file,
    input_names=["pixel_values", "input_ids", "attention_mask"],
    output_names=["pred_masks", "pred_boxes", "pred_logits"],
    opset_version=17,
    external_data=True
)


import onnxruntime as ort

session = ort.InferenceSession(
    output_file,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

# 获取onnx的输入输出各种信息
input_info = session.get_inputs()
output_info = session.get_outputs()

print("input_info:")
for info in input_info:
    print(info.name, info.shape, info.type)

print("\noutput_info:")
for info in output_info:
    print(info.name, info.shape, info.type)

"""
input_info:
pixel_values [1, 3, 1008, 1008] tensor(float)
input_ids [1, 32] tensor(int64)
attention_mask [1, 32] tensor(int64)

output_info:
pred_masks [1, 200, 288, 288] tensor(float)
pred_boxes [1, 200, 4] tensor(float)
pred_logits [1, 200] tensor(float)
"""

# print("over")


