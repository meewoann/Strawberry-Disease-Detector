import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from ultralytics import YOLO

class FruitRipenessDetector:
    def __init__(self, model_path='../runs/segment/train_improved/weights/best.pt', conf_threshold=0.5, tau=50):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.tau = tau

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
        class_ids = boxes.cls.cpu().numpy().astype(int)
        scores = boxes.conf.cpu().numpy()

        h_orig, w_orig = image.shape[:2]
        h_resize, w_resize = resized_image.shape[:2]
        scale_x = w_orig / w_resize
        scale_y = h_orig / h_resize


        crop_imgs_with_status = []

        for i, mask in enumerate(masks.data.cpu().numpy()):
            if scores[i] <= self.conf_threshold:
                continue

            box = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = box
            x1_orig, y1_orig = int(x1 * scale_x), int(y1 * scale_y)
            x2_orig, y2_orig = int(x2 * scale_x), int(y2 * scale_y)

            fruit_crop = image[y1_orig:y2_orig, x1_orig:x2_orig]
            mask_resized = cv2.resize(mask.astype(np.uint8), (x2_orig - x1_orig, y2_orig - y1_orig))
            mask_bin = mask_resized > 0.5

            fruit_pixels = fruit_crop[mask_bin]

            if fruit_pixels.size == 0:
                variance = 0
            else:
                # Chuyển fruit_pixels từ BGR sang HSV
                hsv_pixels = cv2.cvtColor(fruit_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
                hue_values = hsv_pixels[:, 0].astype(np.float32)  # Lấy kênh Hue

                mean_hue = np.mean(hue_values)
                variance = np.mean((hue_values - mean_hue) ** 2)

            ripeness = "Ripe" if variance < self.tau else "Unripe/Defective"

            print(f"Fruit {i+1} ({self.model.names[class_ids[i]]}): Confidence={scores[i]:.2f}, Variance Hue={variance:.1f}, Status={ripeness}")

            # Crop ảnh resize nhỏ để gửi về frontend
            crop_small = cv2.resize(fruit_crop, (150, 150))  # resize vừa đủ để hiển thị
            crop_b64 = self.img_to_base64(crop_small)

            crop_imgs_with_status.append({
                'image': crop_b64,
                'status': ripeness,
                'label': f"{self.model.names[class_ids[i]]} ({scores[i]:.2f})"
            })


            color = (0, 255, 0) if ripeness == "Ripe" else (0, 0, 255)
            cv2.rectangle(resized_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f'{self.model.names[class_ids[i]]} {scores[i]:.2f} VarHue:{variance:.1f} {ripeness}'
            cv2.putText(resized_image, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        full_img_b64 = self.img_to_base64(resized_image)

        return full_img_b64, crop_imgs_with_status
