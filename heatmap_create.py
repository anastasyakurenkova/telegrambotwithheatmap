import cv2
import numpy as np


def update_heatmap(heatmap, detections, frame_shape, target_classes):
    # Обновление карты плотности на основании детекций
    for detect in detections:
        class_id = int(detect.cls)
        if class_id not in target_classes:
            continue

        # Извлечение и форматирование координат рамки
        if detect.xyxy.shape[0] == 1:
            x1, y1, x2, y2 = map(int, detect.xyxy[0].tolist())
            # Добавление информации на карту плотности
            heatmap[y1:y2, x1:x2] += 1

    return heatmap


def generate_heatmap_overlay(frame, heatmap):
    # Нормализация карты плотности
    normalized_heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    normalized_heatmap = np.uint8(normalized_heatmap)

    # Применение цветовой карты
    color_heatmap = cv2.applyColorMap(normalized_heatmap, cv2.COLORMAP_JET)

    # Наложение карты плотности на исходный кадр
    overlay_frame = cv2.addWeighted(frame, 0.6, color_heatmap, 0.4, 0)

    return overlay_frame