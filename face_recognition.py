# face_recognition.py

import cv2
import numpy as np
import os
import datetime
import json
import argparse

USERS_FILE = "users.json"

def load_users():
    """
    Считываем users.json и конвертируем ключи (ID) из str в int,
    чтобы они совпадали с типом predicted_id, который возвращает recognizer.predict().
    """
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Преобразуем ключи словаря к int, а значения (имена) оставляем нетронутыми
    user_dict = {}
    for k, v in data.items():
        try:
            user_dict[int(k)] = v
        except ValueError:
            # если почему-то ключ не число, можно пропустить или обработать иначе
            pass
    return user_dict

# Загружаем словарь имен (ID -> Имя)
USER_NAMES = load_users()

def process_video_stream(video_source, delay):
    # Загружаем каскад Хаара
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # Инициализируем распознаватель и подгружаем обученную модель
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    if os.path.exists("trained_faces.yml"):
        recognizer.read("trained_faces.yml")
    else:
        print("Ошибка: не найден файл 'trained_faces.yml'. Сначала запустите тренеровку (trainer.py).")
        return

    # Открываем видеопоток
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Не удалось открыть видеопоток {video_source}. Проверьте его доступность и права.")
        return

    # Файл для логов (открываем в режиме добавления)
    log_file = open("face_log.txt", "a", encoding="utf-8")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не удалось получить кадр с видеопотока.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Вырезаем лицо из кадра
            roi_gray = gray[y:y+h, x:x+w]
            resized_face = cv2.resize(roi_gray, (256, 256),interpolation=cv2.INTER_CUBIC)

            # Пытаемся распознать
            try:
                predicted_id, confidence = recognizer.predict(resized_face)
            except cv2.error as e:
                print(f"Ошибка при распознавании лица: {e}")
                continue

            # Преобразуем confidence в некий "процент уверенности"
            # (чем меньше confidence у LBPH, тем выше реальная уверенность)
            confidence_text = max(0, min(100, int(100 - confidence)))

            # Проверяем, есть ли такой ID в словаре и достаточно ли высокий "процент"
            if predicted_id in USER_NAMES and confidence_text > 30:
                name = USER_NAMES[predicted_id]
                label = f"{name} ({confidence_text}%)"
                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_file.write(f"{now} - Распознан: {name}, Уверенность: {confidence_text}%\n")
                color = (0, 255, 0)
            else:
                label = f"Неопознанный ({confidence_text}%)"
                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_file.write(f"{now} - Неопознанный, Уверенность: {confidence_text}%\n")
                color = (0, 0, 255)

            # Рисуем рамку и подпись на кадре
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2)

        cv2.imshow("Face Recognition", frame)

        # Нажмите 'q' для выхода
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    # Закрываем ресурсы
    cap.release()
    cv2.destroyAllWindows()
    log_file.close()

def main():
    parser = argparse.ArgumentParser(description="Face Recognition")
    parser.add_argument("--stream", type=str, help="URL of the video stream")
    parser.add_argument("--delay", type=int, default=30, help="Delay between frames in milliseconds")
    args = parser.parse_args()

    # Загружаем каскад Хаара
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    recognizer_type = input("Eigen|Fisher|LBPH\n")
    # Инициализируем распознаватель и подгружаем обученную модель
    if recognizer_type == 'Eigen':
        recognizer = cv2.face.EigenFaceRecognizer_create(num_components=1, threshold=11500)
    elif recognizer_type == 'Fisher':
        recognizer = cv2.face.FisherFaceRecognizer_create(num_components=2, threshold=11500)
    elif recognizer_type == 'LBPH':
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    else:
        print("не указан тип распознавателя")
        return
    if os.path.exists("trained_faces.yml"):
        recognizer.read("trained_faces.yml")
    else:
        print("Ошибка: не найден файл 'trained_faces.yml'. Сначала запустите тренеровку (trainer.py).")
        return

    if args.stream:
        process_video_stream(args.stream, args.delay)
    else:
        # Открываем веб-камеру
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Не удалось открыть камеру. Проверьте её доступность и права.")
            return

        # Файл для логов (открываем в режиме добавления)
        log_file = open("face_log.txt", "a", encoding="utf-8")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Не удалось получить кадр с камеры.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                # Вырезаем лицо из кадра
                roi_gray = gray[y:y+h, x:x+w]
                resized_face = cv2.resize(roi_gray, (273, 273),interpolation=cv2.INTER_CUBIC)
                # Пытаемся распознать
                try:
                    predicted_id, confidence = recognizer.predict(resized_face)
                except cv2.error as e:
                    print(f"Ошибка при распознавании лица: {e}")
                    continue

                # Преобразуем confidence в некий "процент уверенности"
                # (чем меньше confidence у LBPH, тем выше реальная уверенность)
                confidence_text = max(0, min(100, int(100 - confidence)))
                
                # Проверяем, есть ли такой ID в словаре и достаточно ли высокий "процент"
                if predicted_id in USER_NAMES and (confidence_text > 30 or recognizer_type != "LBPH"):
                    print(confidence)
                    
                    name = USER_NAMES[predicted_id]
                    label = f"{name} ({confidence_text}%)"
                    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_file.write(f"{now} - Распознан: {name}, Уверенность: {confidence_text}%\n")
                    color = (0, 255, 0)
                else:
                    label = f"Неопознанный ({confidence_text}%)"
                    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_file.write(f"{now} - Неопознанный, Уверенность: {confidence_text}%\n")
                    color = (0, 0, 255)

                # Рисуем рамку и подпись на кадре
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-10),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2)

            cv2.imshow("Face Recognition", frame)

            # Нажмите 'q' для выхода
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Закрываем ресурсы
        cap.release()
        cv2.destroyAllWindows()
        log_file.close()

if __name__ == "__main__":
    main()