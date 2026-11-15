
from picamera2 import Picamera2
from face_identifier import FaceIdentifier
from database import FaceDatabase
from http.server import BaseHTTPRequestHandler, HTTPServer
import cv2

# Initialize YuNet + SFace
identifier = FaceIdentifier()

# Load known facial embeddings from SQLite
db = FaceDatabase()
known_faces = db.load_embeddings()

print(f"Loaded {len(known_faces)} known face embeddings")

# Minimum cosine similarity for recognition
THRESHOLD = 0.50

# Initialize Raspberry Pi Camera Module 3
camera = Picamera2()

camera.configure(
    camera.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
)

camera.start()


def process_frame(frame):

    annotated_frame = frame.copy()

    # STEP 2: Detect faces and extract embeddings
    faces = identifier.get_face_data(frame)

    # STEP 3: Identify each face
    for embedding, (x, y, w, h) in faces:
        best_score = -1.0
        best_name = "UNKNOWN"

        for name, stored_embedding in known_faces:
            score = identifier.recognizer.match(
                stored_embedding,
                embedding,
                cv2.FaceRecognizerSF_FR_COSINE
            )

            if score > best_score:
                best_score = score
                best_name = name

        # STEP 4: Apply recognition threshold
        if best_score >= THRESHOLD:
            label = f"{best_name} ({best_score:.3f})"
            color = (0, 255, 0)
        else:
            label = "UNKNOWN"
            color = (0, 0, 255)

        # Draw face bounding box
        cv2.rectangle(
            annotated_frame,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

        # Position label above or below face
        label_y = y - 10 if y >= 30 else y + h + 25

        cv2.putText(
            annotated_frame,
            label,
            (max(x, 0), label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    return annotated_frame


class CameraHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header(
            "Content-Type",
            "multipart/x-mixed-replace; boundary=frame"
        )
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        print("Browser connected")

        try:
            while True:
                # Capture camera frame
                frame = camera.capture_array()

                # Run YOLO + facial recognition
                annotated_frame = process_frame(frame)

                # Encode processed frame as JPEG
                success, jpeg = cv2.imencode(
                    ".jpg",
                    annotated_frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 75]
                )

                if not success:
                    continue

                # Send frame to browser
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(
                    f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                )
                self.wfile.write(jpeg.tobytes())
                self.wfile.write(b"\r\n")

        except (BrokenPipeError, ConnectionResetError):
            print("Browser disconnected")


server = HTTPServer(("127.0.0.1", 8000), CameraHandler)

print("YuNet + SFace recognition running on port 8000")

try:
    server.serve_forever()
except KeyboardInterrupt:
    print("Stopping camera stream...")
finally:
    server.server_close()
    camera.stop()
    camera.close()
    db.close()