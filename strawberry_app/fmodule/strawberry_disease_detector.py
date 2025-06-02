import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from ultralytics import YOLO

class StrawberryDiseaseDetector:
    def __init__(self, model_paths='../runs/segment/yolov8_segment_basic3/weights/best.pt', conf_threshold=0.5):
        self.model = YOLO(model_paths)
        self.conf_threshold = conf_threshold

    def img_to_base64(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        buff = BytesIO()
        pil_img.save(buff, format="JPEG")
        return base64.b64encode(buff.getvalue()).decode('utf-8')

    def process_image(self, image):
        resized_image = cv2.resize(image, (768, 768))
        results = self.model(resized_image)

        masks = results[0].masks
        boxes = results[0].boxes
        if masks is None:
            print("Warning: No masks detected!")
            # Bạn có thể trả về ảnh resized không vẽ gì và crops rỗng
            full_img_b64 = self.img_to_base64(resized_image)
            return full_img_b64, []

        class_ids = boxes.cls.cpu().numpy().astype(int)
        scores = boxes.conf.cpu().numpy()

        h_orig, w_orig = image.shape[:2]
        h_resize, w_resize = resized_image.shape[:2]
        scale_x = w_orig / w_resize
        scale_y = h_orig / h_resize

        crop_imgs = []

        for idx, name in self.model.names.items():
            print(f"Class ID {idx}: {name}")

        for i, mask in enumerate(masks.data.cpu().numpy()):
            if scores[i] < self.conf_threshold:
                continue

            box = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = box
            x1_orig, y1_orig = int(x1 * scale_x), int(y1 * scale_y)
            x2_orig, y2_orig = int(x2 * scale_x), int(y2 * scale_y)

            # Crop ảnh gốc dâu bị bệnh
            crop = image[y1_orig:y2_orig, x1_orig:x2_orig]
            crop_resized = cv2.resize(crop, (150, 150))
            crop_b64 = self.img_to_base64(crop_resized)

            class_name = self.model.names[class_ids[i]]
            confidence = scores[i]
            class_id = class_ids[i]

            crop_imgs.append({
                'image': crop_b64,
                'label': class_name,
                'confidence': confidence
            })
            # Vẽ bbox và mask trên ảnh resized
            color = (0, 0, 255)  # màu đỏ cho dâu bị bệnh
            cv2.rectangle(resized_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f'{class_name} {confidence:.2f}'
            cv2.putText(resized_image, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Vẽ mask với màu đỏ trong suốt
            mask_img = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            overlay = resized_image.copy()
            cv2.drawContours(overlay, contours, -1, color, thickness=cv2.FILLED)
            alpha = 0.4
            cv2.addWeighted(overlay, alpha, resized_image, 1 - alpha, 0, resized_image)

        full_img_b64 = self.img_to_base64(resized_image)

        return full_img_b64, crop_imgs
