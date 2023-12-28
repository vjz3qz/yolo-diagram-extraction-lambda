import json
import base64
from io import BytesIO
from PIL import Image
import onnxruntime as ort
import cv2
import numpy as np
import torch
from yolov5.utils.general import non_max_suppression, xyxy2xywh
from PIL import ImageDraw, ImageFont

# Initialize ONNX runtime session
ort_session = ort.InferenceSession("models/yolov5.onnx")

def lambda_handler(event=None, context=None):
    # Decode image
    # body = json.loads(event['body'])
    # img_b64 = body['image']
    # img = Image.open(BytesIO(base64.b64decode(img_b64)))
    img = Image.open('images/diagram.jpeg')
    
    # Preprocess the image
    img_np = np.array(img)
    resized = cv2.resize(img_np, (640, 640)).astype(np.float32) / 255.
    resized = resized.transpose((2, 0, 1))
    resized = np.expand_dims(resized, axis=0)

    # Run inference
    ort_inputs = {ort_session.get_inputs()[0].name: resized}
    ort_outs = ort_session.run(None, ort_inputs)
    print([o.shape for o in ort_outs])

    # Post-process the output
    output = torch.from_numpy(np.asarray(ort_outs))
    out = non_max_suppression(output, conf_thres=0.2, iou_thres=0.5)[0]

    # Draw bounding boxes and class labels
    draw = ImageDraw.Draw(img)
    for detection in out:
        x1, y1, x2, y2, conf, class_id = detection
        # Draw rectangle
        draw.rectangle(((x1, y1), (x2, y2)), outline="red", width=2)
        # Optionally add text (like class name and confidence)
        draw.text((x1, y1), f"Class {class_id}: {conf:.2f}", fill="red")

    # Encode and return the image
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    # save image to the current directory
    img.save('images/diagram_yolo.jpeg')
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return {
        "statusCode": 200,
        "body": json.dumps({
            "image": img_str,
            "detections": out.tolist(),
            "message": "Success"
        })
    }

if __name__ == "__main__":
    print(lambda_handler())