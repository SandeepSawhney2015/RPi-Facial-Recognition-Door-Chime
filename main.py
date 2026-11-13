
import cv2
import sys
from ultralytics import YOLO
from face_identifier import FaceIdentifier
from database import FaceDatabase

# Initialize models
model = YOLO("yolo11s.pt")
identifier = FaceIdentifier()

# Load known faces from SQLite
db = FaceDatabase()
known_faces = db.load_embeddings()

THRESHOLD = 0.50

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
        annotated_frame = results[0].plot()

        # STEP 2: Detect faces and get locations
        faces = identifier.get_face_data(frame)

        # STEP 3: Identify each detected face
        for embedding, (x, y, w, h) in faces:
            best_score = -1.0
            best_name = "UNKNOWN"
            # Compare against stored face embeddings
            for name, stored_embedding in known_faces:
                score = identifier.recognizer.match(
                    stored_embedding,
                    embedding,
                    cv2.FaceRecognizerSF_FR_COSINE)

                if score > best_score:
                    best_score = score
                    best_name = name

            # STEP 4: Determine identity
            if best_score >= THRESHOLD:
                label = f"{best_name} ({best_score:.3f})"
                color = (0, 255, 0)
            else:
                label = "UNKNOWN"
                color = (0, 0, 255)

            # Draw bounding box around face
            cv2.rectangle(
                annotated_frame,
                (x, y),
                (x + w, y + h),
                color,
                2
            )

            # Draw identity above or below face
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

        # STEP 5: Display camera feed
        cv2.imshow(
            "Door Chime - AI Detection",
            annotated_frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    db.close()
