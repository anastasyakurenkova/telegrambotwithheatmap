from ultralytics import YOLO
import cv2
import numpy as np


# Ваши действия по завершении цикла: сохранение тепловой карты
# Сохранение итоговой тепловой карты, наложенной на последний кадр видео
def save_final_heatmap_overlay(last_frame, heatmap):
    # Нормализация тепловой карты
    normalized_heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    normalized_heatmap = np.uint8(normalized_heatmap)

    # Применение цветовой карты
    color_heatmap = cv2.applyColorMap(normalized_heatmap, cv2.COLORMAP_JET)

    # Изменение размера карты цветового тепла, если требуется
    if last_frame.shape[:2] != color_heatmap.shape[:2]:
        color_heatmap = cv2.resize(color_heatmap, (last_frame.shape[1], last_frame.shape[0]))

    # Наложение тепловой карты на последний кадр
    final_overlay = cv2.addWeighted(last_frame, 0.6, color_heatmap, 0.4, 0)

    # Сохранение итогового изображения
    cv2.imwrite('final_overlay.png', final_overlay)

# Инициализация переменной для последнего кадра
last_frame = None
def update_heatmap(heatmap, detections, target_classes, frame_diff):
    # Обновление карты плотности только для движущихся объектов
    for detect in detections:
        class_id = int(detect.cls)
        if class_id not in target_classes:
            continue

        # Извлечение и форматирование координат рамки
        if detect.xyxy.shape[0] == 1:
            x1, y1, x2, y2 = map(int, detect.xyxy[0].tolist())
            # Проверка, есть ли движение в области рамки
            if np.sum(frame_diff[y1:y2, x1:x2]) > 0:
                heatmap[y1:y2, x1:x2] += 1

    return heatmap

def generate_heatmap_overlay(frame, heatmap):
    # Нормализация карты плотности
    normalized_heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    normalized_heatmap = np.uint8(normalized_heatmap)

    # Применение цветовой карты
    color_heatmap = cv2.applyColorMap(normalized_heatmap, cv2.COLORMAP_JET)

    # Изменение размера карты цветового тепла, если требуется
    if frame.shape[:2] != color_heatmap.shape[:2]:
        color_heatmap = cv2.resize(color_heatmap, (frame.shape[1], frame.shape[0]))

    # Наложение карты плотности на исходный кадр
    overlay_frame = cv2.addWeighted(frame, 0.6, color_heatmap, 0.4, 0)

    return overlay_frame

# Загрузка модели YOLOv8
model = YOLO('yolov8n.pt')  # Обновите путь при необходимости

# Открытие видеофайла
video_path = r"C:\projectcomputergraphics\source\455409_Brussels_Bruxelles_1280x720.mp4"
cap = cv2.VideoCapture(video_path)

# Определение индексов классов для отслеживания (например: автомобили)
target_classes = [0, 1, 2, 3, 5, 7]

# Получаем размеры кадра и инициализируем карту плотности
ret, frame1 = cap.read()
if not ret:
    print("Не удалось получить кадр из видео.")
    cap.release()
    exit()

height, width = frame1.shape[:2]
heatmap = np.zeros((height, width), dtype=np.float32)

# Настройка записи выходных видео
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
detected_video_writer = cv2.VideoWriter('detected_cars.mp4', fourcc, 30, (width, height))
heatmap_video_writer = cv2.VideoWriter('heatmap_video.mp4', fourcc, 30, (width, height))

while cap.isOpened():
    ret, frame2 = cap.read()
    if not ret:
        break
    # Сохранение текущего кадра как последнего видимого
    last_frame = frame2.copy()
    # Вычисление разности между кадрами для обнаружения движения
    frame_diff = cv2.absdiff(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY),
                              cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY))
    _, frame_diff = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

    # Обновление предыдущего кадра
    frame1 = frame2.copy()

    # Обработка текущего кадра
    frame_blurred = cv2.GaussianBlur(frame2, (5, 5), 0)
    frame_processed = cv2.convertScaleAbs(frame_blurred, alpha=1.5, beta=0)

    # Проведение детекции
    results = model(frame_processed, conf=0.15)

    # Обновление карты плотности для каждого результата
    for result in results:
        heatmap = update_heatmap(heatmap, result.boxes, target_classes, frame_diff)
        # Рисуем рамку вокруг детектированных объектов
        for box in result.boxes:
            class_id = int(box.cls)
            if class_id in target_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cv2.rectangle(frame2, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Создание карты плотности поверх видео
    output_frame = generate_heatmap_overlay(frame2, heatmap)

    # Запись кадров с детекцией и тепловой картой в видеофайлы
    detected_video_writer.write(frame2)
    heatmap_video_writer.write(output_frame)

    # Отображение результата
    cv2.imshow("Detection", output_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Прерывание по нажатию 'q'
        break

# После завершения обработки видео, сохранить наложенную тепловую карту
if last_frame is not None:
    save_final_heatmap_overlay(last_frame, heatmap)
# Освобождение ресурсов
cap.release()
detected_video_writer.release()
heatmap_video_writer.release()
cv2.destroyAllWindows()










