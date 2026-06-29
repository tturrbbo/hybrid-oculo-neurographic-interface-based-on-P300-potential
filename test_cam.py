import cv2
import numpy as np

print(f"OpenCV version: {cv2.__version__}")

# Проверяем индексы 0-5
for i in range(5):
    print(f"\nПроверяю индекс {i}...")
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Явно указываем DirectShow

    if cap.isOpened():
        print(f"  ✅ Камера {i} открылась!")
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"  ✅ Кадр получен! Размер: {frame.shape}")
        else:
            print(f"  ❌ Не удалось прочитать кадр")
        cap.release()
    else:
        print(f"  ❌ Не удалось открыть камеру {i}")