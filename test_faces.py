
import cv2
import sys
from ultralytics import YOLO
from face_identifier import FaceIdentifier
from database import FaceDatabase

# Initialize YOLO object detection
model = YOLO("yolo11n.pt")

# Initialize face detection and recognition
identifier = FaceIdentifier()

# Open SQLite database and load known embeddings
db = FaceDatabase()
known_faces = db.load_embeddings()

# Minimum cosine similarity required for recognition
THRESHOLD = 0.50

# Open MacBook webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not open camera!", file=sys.stderr)
    db.close()
    sys.exit(1)

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Failed to read frame!", file=sys.stderr)
            break

        # STEP 1: YOLO object detection
        results = model(frame, verbose=False)

        # Draw YOLO bounding boxes
        annotated_frame = results[0].plot()

        # STEP 2: Detect faces and generate embeddings
        embeddings = identifier.get_embeddings(frame)

        # STEP 3: Identify detected faces
        if not embeddings:
            cv2.putText(
                annotated_frame,
                "No face detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        for i, embedding in enumerate(embeddings):

            best_score = -1.0
            best_name = "UNKNOWN"

            # Compare current face against stored faces
            for name, stored_embedding in known_faces:
                score = identifier.recognizer.match(
                    stored_embedding,
                    embedding,
                    cv2.FaceRecognizerSF_FR_COSINE
                )

                if score > best_score:
                    best_score = score
                    best_name = name

            # STEP 4: Determine identity
            if best_score >= THRESHOLD:
                label = f"{best_name} ({best_score:.3f})"
                color = (0, 255, 0)  # Green
            else:
                label = "UNKNOWN"
                color = (0, 0, 255)  # Red

            # Display recognized name on screen
            cv2.putText(
                annotated_frame,
                label,
                (20, 40 + i * 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        # STEP 5: Display video
        cv2.imshow("Door Chime - AI Detection", annotated_frame)

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    db.close()
