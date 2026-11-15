from picamera2 import Picamera2
from http.server import BaseHTTPRequestHandler, HTTPServer
import cv2

# Initialize Raspberry Pi Camera Module 3
camera = Picamera2()

camera.configure(
    camera.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
)

camera.start()


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

        try:
            while True:
                # Capture frame from Pi camera
                frame = camera.capture_array()

                # Encode frame as JPEG
                success, jpeg = cv2.imencode(".jpg", frame)

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
            print("Client disconnected")


# Listen locally; VS Code forwards the port over SSH
server = HTTPServer(("127.0.0.1", 8000), CameraHandler)

print("Camera streaming on port 8000")

try:
    server.serve_forever()
except KeyboardInterrupt:
    print("Stopping camera stream...")
finally:
    server.server_close()
    camera.stop()
