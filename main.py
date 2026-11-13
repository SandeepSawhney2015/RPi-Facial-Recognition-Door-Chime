
import cv2
import sys
from ultralytics import YOLO
from face_identifier import FaceIdentifier
from database import FaceDatabase

# Initialize models
model = YOLO("yolo11n.pt")
identifier = FaceIdentifier()

# Initialize SQLite database
db = FaceDatabase()
known_faces = db.load_embeddings()

THRESHOLD = 0.50

# Name used when enrolling
enrollment_name = None

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

        # STEP 2: Detect faces and get their locations
        faces = identifier.get_face_data(frame)

        # STEP 3: Identify each detected face
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

            # Determine identity
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

            # Position label beside the detected face
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

        # STEP 4: Display enrollment instructions
        instructions = "N: Set Name | E: Enroll | Q: Quit"

        cv2.putText(
            annotated_frame,
            instructions,
            (20, annotated_frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        if enrollment_name:
            cv2.putText(
                annotated_frame,
                f"Enrolling: {enrollment_name}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        cv2.imshow("Door Chime - AI Detection", annotated_frame)

        key = cv2.waitKey(1) & 0xFF

        # STEP 5: Set enrollment name
        if key == ord("n"):
            name = input("Enter person's name: ").strip().lower()

            if name:
                enrollment_name = name
                print(f"Enrollment name set to: {name}")
            else:
                print("Name cannot be empty!")

        # STEP 6: Enroll a person
        elif key == ord("e"):

            if enrollment_name is None:
                print("Press N to set a name first!")

            elif len(faces) == 1:
                embedding, _ = faces[0]

                db.add_embedding(
                    enrollment_name,
                    embedding
                )

                # Update in-memory database
                known_faces.append(
                    (enrollment_name, embedding)
                )

                print(
                    f"Embedding saved for {enrollment_name}!"
                )

            else:
                print(
                    "Enrollment requires exactly one face!"
                )

        elif key == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    db.close()
