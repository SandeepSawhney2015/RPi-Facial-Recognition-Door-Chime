
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

# Name of person currently being enrolled
enrollment_name = None

# Track number of samples enrolled this session
enrollment_count = 0

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

        # STEP 2: Detect faces and generate embeddings
        embeddings = identifier.get_embeddings(frame)

        # STEP 3: Recognize each face
        for i, embedding in enumerate(embeddings):
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

            if best_score >= THRESHOLD:
                label = f"{best_name} ({best_score:.3f})"
                color = (0, 255, 0)
            else:
                label = "UNKNOWN"
                color = (0, 0, 255)

            cv2.putText(
                annotated_frame,
                label,
                (20, 50 + i * 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        # STEP 4: Display enrollment status
        status = (
            f"Enrolling: {enrollment_name} ({enrollment_count})"
            if enrollment_name
            else "No enrollment name selected"
        )

        cv2.putText(
            annotated_frame,
            status,
            (20, annotated_frame.shape[0] - 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.putText(
            annotated_frame,
            "N: Set Name | E: Enroll | Q: Quit",
            (20, annotated_frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow("Face Enrollment", annotated_frame)

        key = cv2.waitKey(1) & 0xFF

        # STEP 5: Select person to enroll
        if key == ord("n"):
            name = input("Enter person's name: ").strip().lower()

            if name:
                enrollment_name = name
                enrollment_count = 0
                print(f"Now enrolling: {enrollment_name}")
            else:
                print("Name cannot be empty!")

        # STEP 6: Save face embedding
        elif key == ord("e"):
            if enrollment_name is None:
                print("Press N to set a name first!")

            elif len(embeddings) != 1:
                print("Enrollment requires exactly one face!")

            else:
                embedding = embeddings[0]

                # Save the embedding to SQLite
                db.add_embedding(enrollment_name, embedding)

                # Also update the in-memory collection
                known_faces.append((enrollment_name, embedding))

                enrollment_count += 1

                print(
                    f"Saved embedding #{enrollment_count} "
                    f"for {enrollment_name}!"
                )

        elif key == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    db.close()
