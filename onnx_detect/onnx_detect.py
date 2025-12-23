from PIL import Image
import requests


import onnxruntime as ort
import numpy as np
import cv2
# json re

# 当前项目目录加载为环境变量
import os
os.environ["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


try:
    from onnx_detect.image_preprocess import preprocess_image
    from onnx_detect.tokenizer import SimpleCLIPBPETokenizer
    from onnx_detect.detect_postprocess import process_sam3_results, draw_sam3_results
except:
    from image_preprocess import preprocess_image
    from simplify_tokenizer import SimpleCLIPBPETokenizer
    from detect_postprocess import process_sam3_results, draw_sam3_results



def main():
    image_url = "./onnx_export/zidane.jpg"
    onnx_file_path = "models/onnx_detect/sam3.onnx"

    image = cv2.imread(image_url)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    session = ort.InferenceSession(
        onnx_file_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    vocab_file = "models/vocab.json"
    merges_file = "models/merges.txt"
    prompt = "person"

    tokenizer = SimpleCLIPBPETokenizer(
        vocab_file=vocab_file,
        merges_file=merges_file,
        max_length=32,
        bos_token_id=49406,
        eos_token_id=49407,
        bpe_vocab_size=49152,
    )

    processed_image = preprocess_image(image)
    
    ids, mask = tokenizer.encode(prompt)
    input_ids = np.array(ids, dtype=np.int64).reshape(1, -1)
    attention_mask = np.array(mask, dtype=np.int64).reshape(1, -1)

    input_dict = {
        "pixel_values": processed_image.astype("float32"),
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    outputs = session.run(None, input_dict)


    image = np.array(image)
    results = process_sam3_results(
        outputs,
        img_h=image.shape[0],
        img_w=image.shape[1],
        score_thr=0.6,
        mask_thr=0.5,
        max_inst=30,
        boxes_normalized=True,
    )
    vis_img = draw_sam3_results(image, results)
    
    cv2.imwrite("vis.png", vis_img)


if __name__ == "__main__":
    main()
