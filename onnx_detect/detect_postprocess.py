import cv2
import numpy as np
import random

def process_sam3_results(
    outputs,
    img_h,
    img_w,
    score_thr=0.4,
    mask_thr=0.5,
    max_inst=30,
    boxes_normalized=True,
):
    """
    处理 SAM3 输出，返回结构化的检测结果列表。
    
    outputs: [pred_masks, pred_boxes, pred_logits]
    img_h, img_w: 原图高宽，用于缩放 mask 和 box
    """
    pred_masks, pred_boxes, pred_logits = outputs

    # 去掉 batch 维
    if pred_masks.ndim == 4:
        pred_masks = pred_masks[0]      # [N, Hm, Wm]
    if pred_boxes.ndim == 3:
        pred_boxes = pred_boxes[0]      # [N, 4]
    if pred_logits.ndim == 2:
        pred_logits = pred_logits[0]    # [N]

    pred_masks  = pred_masks.astype(np.float32)
    pred_boxes  = pred_boxes.astype(np.float32)
    pred_logits = pred_logits.astype(np.float32)

    N, Hm, Wm = pred_masks.shape

    # logits -> scores
    scores = 1.0 / (1.0 + np.exp(-pred_logits))  # [N]

    # 排序 + 阈值 + topk
    indices = list(range(N))
    indices = sorted(indices, key=lambda i: float(scores[i]), reverse=True)
    indices = [i for i in indices if float(scores[i]) >= score_thr]
    indices = indices[:max_inst]

    results = []
    for i in indices:
        mask  = pred_masks[i]          # [Hm, Wm]
        score = float(scores[i])
        box   = pred_boxes[i]          # [4]

        # 1) 先把 mask resize 到原图大小
        mask_resized = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        m = (mask_resized > mask_thr).astype(np.uint8)  # [H, W]

        if m.sum() == 0:
            continue

        # 2) 处理框坐标
        x1, y1, x2, y2 = box

        if boxes_normalized:
            x1 = int(x1 * img_w)
            x2 = int(x2 * img_w)
            y1 = int(y1 * img_h)
            y2 = int(y2 * img_h)
        else:
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

        # clamp 一下防止越界
        x1 = max(0, min(x1, img_w - 1))
        x2 = max(0, min(x2, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        y2 = max(0, min(y2, img_h - 1))

        results.append({
            "mask": m,          # [H, W] uint8
            "box": [x1, y1, x2, y2],
            "score": score,
        })
        
    return results

def draw_sam3_results(
    img,
    results,
):
    """
    在原图上画检测结果。
    
    img: 原图 (H, W, 3)
    results: process_sam3_results 返回的列表
    """
    vis_img = img.copy().astype(np.float32)
    img_h, img_w = img.shape[:2]

    for item in results:
        m = item["mask"]
        box = item["box"]
        score = item["score"]

        # 随机颜色
        color = [random.randint(0, 255) for _ in range(3)]

        # 3 通道 mask
        m3 = np.stack([m, m, m], axis=-1)  # [H, W, 3]

        # 在原图上叠加半透明颜色
        vis_img = np.where(
            m3 == 1,
            vis_img * 0.5 + np.array(color, dtype=np.float32) * 0.5,
            vis_img,
        )

        x1, y1, x2, y2 = box

        # 画框
        cv2.rectangle(
            vis_img,
            (x1, y1),
            (x2, y2),
            color,
            2,
            lineType=cv2.LINE_AA,
        )

        # 写分数
        cv2.putText(
            vis_img,
            f"{score:.2f}",
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    vis_img = np.clip(vis_img, 0, 255).astype(np.uint8)
    return vis_img
