import cv2
import numpy as np
from ultralytics import YOLO
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackQueryHandler, CallbackContext
import os
from dotenv import load_dotenv
load_dotenv()
#Загружаем токен для бота
TOKEN_BOT = os.getenv("TOKEN_BOT")

# Инициализируем модель
model = YOLO('yolov8n.pt')

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
def start(update: Update, context: CallbackContext):
    update.message.reply_text('Привет! Отправь сюда видео, которое хочешь обработать.')


def video_handler(update: Update, context: CallbackContext):
    file_id = update.message.video.file_id
    new_file = context.bot.get_file(file_id)
    video_path = new_file.download()

    update.message.reply_text(
        "Выбери, что хочешь сделать с видео, после выбора придётся подождать, программа долго распознаёт нужные элементы:",
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("Карта плотности(heatmap)", callback_data='heatmap')],
            [InlineKeyboardButton("Накопление heatmap видео", callback_data='accumulate')]
        ])
    )

    # Store the video path for further processing
    context.user_data['video_path'] = video_path


def button_handler(update: Update, context: CallbackContext):
    query = update.callback_query
    query.answer()

    video_path = context.user_data.get('video_path')
    if query.data == 'heatmap':
        result_path = process_video(video_path, 'heatmap')
        query.message.reply_photo(open(result_path, 'rb'))  # Send as photo instead
    elif query.data == 'accumulate':
        result_path = process_video(video_path, 'accumulate')
        query.message.reply_video(open(result_path, 'rb'))


def process_video(video_path, mode):
    cap = cv2.VideoCapture(video_path)
    heatmap = None
    last_frame = None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if mode == 'heatmap':
        # Temporary filename for last frame with heatmap
        result_file = 'last_frame_with_heatmap.jpg'
    else:
        result_file = mode + '_result.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(result_file, fourcc, 30, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if mode == 'heatmap' or mode == 'accumulate':
            if heatmap is None:
                heatmap = np.zeros((height, width), dtype=np.float32)

            last_frame = frame.copy()
            results = model(frame)
            for result in results:
                heatmap = update_heatmap(heatmap, result.boxes, [0, 1, 2, 3, 5, 7], frame)

            if mode == 'accumulate':
                overlay_frame = generate_heatmap_overlay(frame, heatmap)
                out.write(overlay_frame)

    if mode == 'heatmap':
        # Generate last frame with its heatmap
        if last_frame is not None:
            final_overlay = generate_heatmap_overlay(last_frame, heatmap)
            cv2.imwrite(result_file, final_overlay)

    cap.release()
    if mode != 'heatmap':
        out.release()

    return result_file


def main():
    updater = Updater(TOKEN_BOT, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.video, video_handler))
    dp.add_handler(CallbackQueryHandler(button_handler))
    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()











