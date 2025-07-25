from django.shortcuts import render, redirect
from .forms import RegForm
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm


# Create your views here.

@login_required
def main_page(request):
    return render(request,
                  'main.html')

def registration(request):
    if request.method == 'POST':
        reg = UserCreationForm(request.POST)
        if reg.is_valid():
            user = reg.save()
            login(request, user)
            return redirect(main_page)
    else:
        reg = RegForm()
    return render(request, "registration.html", {"form": reg})


import threading
import queue
import cv2
from ultralytics import YOLO
from django.http import StreamingHttpResponse

model = YOLO("best.pt").to("cuda")
model.fuse()

raw_frames = queue.Queue(maxsize=10)
jpeg_frames = queue.Queue(maxsize=10)

def video_capture_worker():
    cap = cv2.VideoCapture("http://192.168.0.2:4747/video")
    if not cap.isOpened():
        print("Ошибка: не удалось открыть видеопоток")
        return

    while True:
        success, frame = cap.read()
        if not success:
            continue

        if raw_frames.full():
            try:
                raw_frames.get_nowait()
            except queue.Empty:
                pass

        raw_frames.put(frame)

    cap.release()

def video_processing_worker():
    while True:
        frame = raw_frames.get()
        if frame is None:
            continue

        frame_small = cv2.resize(frame, (320, 320))
        results = model.predict(frame_small, imgsz=320, conf=0.3)[0]

        scale_x = frame.shape[1] / 320
        scale_y = frame.shape[0] / 320

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)

            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = f"{results.names[cls_id]} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        if jpeg_frames.full():
            try:
                jpeg_frames.get_nowait()
            except queue.Empty:
                pass

        jpeg_frames.put(buffer.tobytes())

def gen_frames():
    while True:
        frame = jpeg_frames.get()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

threads_started = False

def video_feed(request):
    global threads_started
    if not threads_started:
        threading.Thread(target=video_capture_worker, daemon=True).start()
        threading.Thread(target=video_processing_worker, daemon=True).start()
        threads_started = True

    return StreamingHttpResponse(gen_frames(),
                                 content_type='multipart/x-mixed-replace; boundary=frame')

