import os
import io
import datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import cv2
from flask import Flask, render_template, request, redirect, Response, jsonify, url_for, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from sort import Sort

# ==== SETUP FLASK DAN DATABASE ====
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///coconut_counts.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S-%f"

# ==== MODEL DATABASE ====
class CoconutCount(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.String(100), nullable=False)
    count = db.Column(db.Integer, nullable=False)

with app.app_context():
    db.create_all()

# ==== VARIABEL GLOBAL ====
count_coconut_out = 0
counts_directory = 'static/processed'
os.makedirs(counts_directory, exist_ok=True)

# ==== DEVICE & MODEL ====
device = torch.device('cpu')
print(f"[INFO] Using device: {device}")

import pathlib
pathlib.PosixPath = pathlib.WindowsPath  # Windows support

model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.to(device)
model.eval()
model.conf = 0.3
model.iou = 0.45

# ==== ROUTES ====

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return redirect(request.url)

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))

        results = model([img])
        results.render()

        boxes = results.pred[0]
        boxes = boxes[boxes[:, 4] > 0.25]
        count_total = boxes.shape[0]

        img_with_labels = Image.fromarray(results.ims[0])
        draw = ImageDraw.Draw(img_with_labels)
        font = ImageFont.load_default()
        draw.text((20, 20), f"Kelapa Terdeteksi: {count_total}", fill="red", font=font)

        for box in boxes:
            x1, y1, x2, y2, conf, cls = box.tolist()
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            draw.ellipse((center_x - 5, center_y - 5, center_x + 5, center_y + 5), fill="red")
            label = model.names[int(cls)]
            draw.text((x1, y1 - 10), label, fill="red", font=font)

        now_time = datetime.datetime.now().strftime(DATETIME_FORMAT)
        img_savename = f"static/{now_time}.png"
        img_with_labels.save(img_savename)
        return redirect("/" + img_savename)

    return render_template('detect.html')

@app.route("/get_counts")
def get_counts():
    return jsonify({'coconut_out': count_coconut_out})

@app.route("/reset_counts", methods=["POST"])
def reset_counts():
    global count_coconut_out
    count_coconut_out = 0
    return jsonify({"message": "Count reset", "count": count_coconut_out})

@app.route("/save_counts", methods=["POST"])
def save_counts():
    global count_coconut_out
    timestamp = datetime.datetime.now().strftime(DATETIME_FORMAT)

    print(f"[DEBUG] Menyimpan count kelapa = {count_coconut_out} pada waktu {timestamp}")  # Tambah ini

    if count_coconut_out == 0:
        return jsonify({"message": "Count masih 0. Tidak disimpan."}), 400

    new_record = CoconutCount(timestamp=timestamp, count=count_coconut_out)
    db.session.add(new_record)
    db.session.commit()
    return jsonify({
        "message": "Berhasil disimpan.",
        "timestamp": timestamp,
        "count": count_coconut_out
    })

@app.route("/load_counts", methods=["GET"])
def load_counts():
    records = CoconutCount.query.order_by(CoconutCount.timestamp.desc()).all()
    data = [{"timestamp": r.timestamp, "count": r.count} for r in records]
    return jsonify(data)

@app.route("/kelapa_chart_data")
def kelapa_chart_data():
    records = CoconutCount.query.order_by(CoconutCount.timestamp).all()
    data = []
    cumulative = 0
    for r in records:
        cumulative += r.count
        data.append({
            "timestamp": r.timestamp,
            "count": r.count,
            "cumulative": cumulative
        })
    return jsonify(data)

@app.route("/riwayat")
def riwayat():
    records_raw = CoconutCount.query.order_by(CoconutCount.timestamp.desc()).all()
    records = []

    for r in records_raw:
        try:
            dt_obj = datetime.datetime.strptime(r.timestamp, "%Y-%m-%d_%H-%M-%S-%f")
            date_str = dt_obj.strftime("%Y-%m-%d")
            time_str = dt_obj.strftime("%H:%M:%S")
        except ValueError:
            date_str = r.timestamp
            time_str = "-"
        
        records.append({
            "date": date_str,
            "time": time_str,
            "count": r.count
        })
    
    return render_template("riwayat.html", records=records)

@app.route("/video")
def video():
    return Response(gen_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/uploadvideo", methods=["GET", "POST"])
def upload_video():
    if request.method == "POST":
        if "video" not in request.files:
            return "No video part", 400
        file = request.files["video"]
        if file.filename == "":
            return "No selected file", 400


        # Pastikan folder static/processed ada
        processed_dir = os.path.join("static", "processed")
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)

        filename = datetime.datetime.now().strftime(DATETIME_FORMAT) + ".mp4"
        filepath = os.path.join(processed_dir, filename)
        file.save(filepath)

        processed_filename = process_uploaded_video(filepath)
        # processed_filename adalah nama file hasil proses, bukan path lengkap

        # Redirect ke URL video processed
        return redirect(url_for('static', filename=f'processed/{processed_filename}'))

    return render_template("uploadvideo.html")

def process_uploaded_video(file_path):
    global count_coconut_out
    count_coconut_out = 0

    cap = cv2.VideoCapture(file_path)
    output_path = file_path.replace(".mp4", "_processed.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    line_position = height // 2
    tracker = Sort(max_age=50, min_hits=5, iou_threshold=0.2)
    tracked_positions = {}

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = model([img])
        results.render()
        img_with_labels = Image.fromarray(np.squeeze(results.ims[0]))
        draw = ImageDraw.Draw(img_with_labels)

        try:
            font = ImageFont.truetype("arialbd.ttf", 30)
        except IOError:
            font = ImageFont.load_default()

        draw.line([(0, line_position), (img_with_labels.width, line_position)], fill="green", width=6)

        dets = []
        for box in results.xyxy[0].tolist():
            x1, y1, x2, y2, conf, cls = box
            if conf > 0.25:
                dets.append([x1, y1, x2, y2, conf])

        dets = np.array(dets)
        tracked_objects = tracker.update(dets)

        for *bbox, track_id in tracked_objects:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            draw.ellipse([(cx - 5, cy - 5), (cx + 5, cy + 5)], fill="red", outline="red")
            draw.text((x1, y1 - 10), f"ID:{int(track_id)}", fill="red", font=font)

            prev_y = tracked_positions.get(int(track_id), None)
            if prev_y is not None:
                if prev_y < line_position <= cy:
                    count_coconut_out += 1

            tracked_positions[int(track_id)] = cy

        text = f"Kelapa Keluar: {count_coconut_out}"
        bbox = font.getbbox(text) if hasattr(font, "getbbox") else (0, 0, *font.getsize(text))
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.text(((img_with_labels.width - text_width) // 2, line_position - text_height - 10),
                  text, fill="blue", font=font)

        frame_bgr = cv2.cvtColor(np.array(img_with_labels), cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    cap.release()
    out.release()
    return os.path.basename(output_path)

@app.route("/riwayatvideo")
def riwayatvideo():
    video_dir = os.path.join("static", "processed")
    videos = []
    if os.path.exists(video_dir):
        for fname in os.listdir(video_dir):
            if fname.endswith(".mp4"):
                timestamp = os.path.getmtime(os.path.join(video_dir, fname))
                videos.append({
                    "filename": fname,
                    "timestamp": datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                })

    # Urutkan terbaru dulu
    videos.sort(key=lambda x: x["timestamp"], reverse=True)
    return render_template("riwayatvideo.html", videos=videos)

@app.route("/cam")
def cam():
    return render_template("cam.html")

def gen_video():
    global count_coconut_out

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    line_position = frame_height // 2
    tracker = Sort(max_age=50, min_hits=5, iou_threshold=0.2)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = model([img_pil], size=320)
        results.render()
        img_result = Image.fromarray(np.squeeze(results.ims[0]))
        draw = ImageDraw.Draw(img_result)
        font = ImageFont.load_default()
        draw.line([(0, line_position), (img_result.width, line_position)], fill="green", width=5)

        dets = []
        for box in results.xyxy[0].tolist():
            x1, y1, x2, y2, conf, cls = box
            if conf > 0.25:
                dets.append([x1, y1, x2, y2, conf])

        dets = np.array(dets)
        tracked_objects = tracker.update(dets)

        for *bbox, track_id in tracked_objects:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            draw.ellipse([(cx - 5, cy - 5), (cx + 5, cy + 5)], fill="red", outline="red")
            draw.text((x1, y1 - 10), f"ID:{int(track_id)}", fill="red", font=font)

            if not hasattr(gen_video, "tracked_positions"):
                gen_video.tracked_positions = {}
            prev_y = gen_video.tracked_positions.get(int(track_id), None)
            if prev_y is not None and prev_y < line_position <= cy:
                count_coconut_out += 1
            gen_video.tracked_positions[int(track_id)] = cy

        draw.text((20, 20), f"Kelapa Keluar: {count_coconut_out}", fill="blue", font=font)
        result_frame = cv2.cvtColor(np.array(img_result), cv2.COLOR_RGB2BGR)
        _, jpeg = cv2.imencode('.jpg', result_frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    cap.release()

if __name__ == "__main__":
    app.run(debug=True, port=5000)