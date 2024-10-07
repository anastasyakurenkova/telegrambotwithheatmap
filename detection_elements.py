from ultralytics import YOLO
import cv2

# Загрузка модели YOLOv8
model = YOLO('yolov8n.pt')  # Обновите путь при необходимости

# Открытие видеофайла
video_path = r"C:\projectcomputergraphics\source\5753_Tokyo_Japan_1280x720.mp4"
cap = cv2.VideoCapture(video_path)

# Определение индексов классов для отслеживания
target_classes = [0, 1, 2, 3, 5, 6, 7]

# Словарь для цветовой раскраски классов
color_mapping = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Предобработка изображения для повышения качества
    # Применяем размытие и коррекцию контраста
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=0)

    # Проведение детекции с низким порогом уверенности
    results = model(frame, conf=0.15)

    # Работа с каждым обнаруженным объектом
    for result in results:
        for detect in result.boxes:
            class_id = int(detect.cls)
            confidence = float(detect.conf)

            # Обработка только целевых классов
            if class_id not in target_classes:
                continue

            # Извлечение и форматирование координат рамки
            if detect.xyxy.shape[0] == 1:
                x1, y1, x2, y2 = map(int, detect.xyxy[0].tolist())
            else:
                continue  # или обработать как-то иначе

            # Получение имени класса
            class_name = result.names[class_id]

            # Назначение уникального цвета классу
            if class_name not in color_mapping:
                color_mapping[class_name] = (int(class_id * 37 % 256),
                                             int(class_id * 57 % 256),
                                             int(class_id * 97 % 256))

            color = color_mapping[class_name]

            # Рисование прямоугольника и добавление текста
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{class_name}: {confidence:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Отображение результата
    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Прерывание по нажатию 'q'
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()








