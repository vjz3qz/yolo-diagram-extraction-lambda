from yolo_onnx.yolov8_onnx import YOLOv8
from ultralytics import YOLO
import json
import base64
from io import BytesIO
from PIL import Image
from PIL import ImageDraw, ImageFont
# Initialize YOLOv8 object detector
yolov8_detector = YOLOv8('./models/best_yolov8n.onnx')

def lambda_handler(event=None, context=None):

    # # get payload
    # body = json.loads(event['body'])

    # # get params
    # img_b64 = body['image']
    body = {
        "image": "base64",
        "size": 640,
        "conf_thres": 0.3,
        "iou_thres": 0.5
    }
    size = body.get('size', 640)
    conf_thres = body.get('conf_thres', 0.3)
    iou_thres = body.get('iou_thres', 0.5)

    # # open image
    # img = Image.open(BytesIO(base64.b64decode(img_b64.encode('ascii'))))
    # get img form images folder
    img = Image.open('./images/diagram.jpeg')
    # infer result
    detections = yolov8_detector(img, size=size, conf_thres=conf_thres, iou_thres=iou_thres)
    # Draw bounding boxes and class labels
    draw = ImageDraw.Draw(img)
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        conf, class_id = detection['score'], detection['class_id']
        # Draw rectangle
        draw.rectangle(((x1, y1), (x2, y2)), outline="red", width=2)
        # Optionally add text (like class name and confidence)
        draw.text((x1, y1), f"Class {class_id}: {conf:.2f}", fill="red")

    # Encode and return the image
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    # save image to the current directory
    img.save('images/diagram_yolo.jpeg')
    # return result
    return {
        "statusCode": 200,
        "body": json.dumps({
            "detections": detections
        }),
    }

if __name__ == '__main__':
    # print(lambda_handler())
    

    model = YOLO('./weights/best_yolov8n.pt') 
    
    results = model('./images/diagram.jpeg', save=True) 