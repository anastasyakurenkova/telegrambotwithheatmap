from ultralytics import YOLO
import cv2
import numpy as np
from heatmap_create import update_heatmap, generate_heatmap_overlay

# Загрузка модели YOLOv8
model = YOLO('yolov8n.pt')  # Обновите путь при необходимости

# Открытие видеофайла
video_path = r"C:\projectcomputergraphics\source\5753_Tokyo_Japan_1280x720.mp4"
cap = cv2.VideoCapture(video_path)

# Определение индексов классов для отслеживания
target_classes = [0, 1, 2, 3, 5, 7]

# Получаем размеры кадра и инициализируем карту плотности
ret, frame = cap.read()
heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Предобработка изображения
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=0)

    # Проведение детекции
    results = model(frame, conf=0.15)

    # Обновление карты плотности
    for result in results:
        heatmap = update_heatmap(heatmap, result.boxes, frame.shape, target_classes)

    # Создание карты плотности поверх видео
    output_frame = generate_heatmap_overlay(frame, heatmap)

    # Отображение результата
    cv2.imshow("Detection", output_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Прерывание по нажатию 'q'
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()







