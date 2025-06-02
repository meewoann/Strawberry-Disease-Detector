from ultralytics import YOLO
import cv2
import numpy as np

# Load model đã train sẵn của bạn
model = YOLO('runs/segment/train_improved/weights/best.pt')

# Đường dẫn ảnh của bạn
image_path = 'berry.v1i.yolov8/train/images/strawberry_2_jpg.rf.02b3540f7347fb51fd22113cc93f866a.jpg'
image = cv2.imread(image_path)  # Ảnh gốc chưa resize

# Resize ảnh về đúng 728x728 (không giữ tỉ lệ) để model dự đoán
resized_image = cv2.resize(image, (768, 768))

# Dự đoán segmentation
results = model(resized_image)

# Lấy mask, bbox, class, score
masks = results[0].masks
boxes = results[0].boxes
class_ids = boxes.cls.cpu().numpy().astype(int)
scores = boxes.conf.cpu().numpy()

conf_threshold = 0.5
tau = 50  # Ngưỡng variance để phân loại chín (cần điều chỉnh)

h_orig, w_orig = image.shape[:2]
h_resize, w_resize = resized_image.shape[:2]
scale_x = w_orig / w_resize
scale_y = h_orig / h_resize

print("Kết quả dự đoán trạng thái chín của các quả:")

for i, mask in enumerate(masks.data.cpu().numpy()):
    if scores[i] <= conf_threshold:
        continue

    color = np.random.randint(0, 255, (3,), dtype=np.uint8)
    mask_img = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(resized_image, contours, -1, color.tolist(), thickness=cv2.FILLED, lineType=cv2.LINE_AA)

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
        fruit_gray = cv2.cvtColor(fruit_pixels.reshape(-1,1,3), cv2.COLOR_BGR2GRAY).flatten()
        mean_p = np.mean(fruit_gray)
        variance = np.mean((fruit_gray - mean_p) ** 2)

    ripeness = "Ripe" if variance < tau else "Unripe/Defective"

    # In ra terminal
    print(f"Quả {i+1} ({model.names[class_ids[i]]}): Confidence={scores[i]:.2f}, Variance={variance:.1f}, Status={ripeness}")

    # Vẽ bbox và label trên ảnh resized
    cv2.rectangle(resized_image, (int(x1), int(y1)), (int(x2), int(y2)), color.tolist(), 2)
    label = f'{model.names[class_ids[i]]} {scores[i]:.2f} Var:{variance:.1f} {ripeness}'
    cv2.putText(resized_image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color.tolist(), 2)

cv2.imshow('Segmented Image with Ripeness', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
