import cv2
import torch
import tkinter as tk
import time

# 載入你的預訓練模型

model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/berto/Desktop/三下/數值方法/軒豪/yolov5s_best.pt') 
# 開啟前鏡頭
cap = cv2.VideoCapture(0)

# 建立一個新的 tkinter 視窗
root = tk.Tk()
root.withdraw()  # 隱藏主視窗

start_time = time.time()

while True:
    # 讀取一幀影像
    ret, frame = cap.read()
    if not ret:
        break

    # 進行物件偵測
    results = model(frame)

    # 繪製偵測結果
    detected_objects = []
    for *box, conf, cls in results.xyxy[0]:
        label = f'{results.names[int(cls)]}: {conf:.2f}'
        detected_objects.append(results.names[int(cls)])
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # 每隔 0.5 秒，輸出偵測到的物件
    if time.time() - start_time >= 0.5:
        print(f"Detected objects: {detected_objects}")
        start_time = time.time()

    # 如果偵測到火災，則顯示警報視窗
    if "fire" in detected_objects:
        alert = tk.Toplevel(root)
        alert.title("警報")
        tk.Label(alert, text="火災警報").pack()
        root.update()

    # 顯示影像
    cv2.imshow('Camera', frame)

    # 如果按下 q 鍵，則跳出迴圈
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源並關閉視窗
cap.release()
cv2.destroyAllWindows()
root.destroy()
