
import onnxruntime as ort
from transformers import Sam3Processor, Sam3Model
import cv2

device = "cpu"
image_url = "./onnx_export/zidane.jpg"
onnx_file_path = "models/onnx_detect/sam3.onnx"

image = cv2.imread(image_url)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



model = Sam3Model.from_pretrained("models").to(device)
processor = Sam3Processor.from_pretrained("models")

model.eval()

inputs = processor(images=image, text="ear", return_tensors="pt").to(device)

pixel_values = inputs["pixel_values"]
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

input_dict = {
    "pixel_values": pixel_values.numpy().astype("float32"),
    "input_ids": input_ids.numpy().astype("int64"),
    "attention_mask": attention_mask.numpy().astype("int64"),
}


session = ort.InferenceSession(
    onnx_file_path,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)


output_dict = session.run(None, input_dict)

pred_masks, pred_boxes, pred_logits = output_dict

# 打印维度
print("pred_masks:", pred_masks.shape)
print("pred_boxes:", pred_boxes.shape)
print("pred_logits:", pred_logits.shape)


"""
pred_masks: (1, 200, 288, 288)
pred_boxes: (1, 200, 4)
pred_logits: (1, 200)
"""