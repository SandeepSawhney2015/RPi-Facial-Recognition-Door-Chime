
import cv2
from face_identifier import FaceIdentifier
from database import FaceDatabase

identifier = FaceIdentifier()
db = FaceDatabase()

# Load existing embeddings from SQLite
known_faces = db.load_embeddings()

# Initial threshold (adjust after testing)
THRESHOLD = 0.50

# Name currently being enrolled
enrollment_name = None

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Camera not available")

try:
    while True:
        success, frame = camera.read()

        if not success:
            break

        embeddings = identifier.get_embeddings(frame)

        # Compare every detected face against our database
        for i, embedding in enumerate(embeddings):

            best_score = -1.0
            best_name = "UNKNOWN"

            # Compare against every saved embedding
            for name, stored in known_faces:
                score = identifier.recognizer.match(
                    stored,
                    embedding,
                    cv2.FaceRecognizerSF_FR_COSINE
                )

                # Update the best matching identity
                if score > best_score:
                    best_score = score
                    best_name = name

            # Determine whether the match is sufficient
            if best_score >= THRESHOLD:
                label = f"{best_name} ({best_score:.3f})"
                color = (0, 255, 0)
            else:
                label = "UNKNOWN"
                color = (0, 0, 255)

            cv2.putText(
                frame,
                label,
                (20, 50 + i * 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        # Display enrollment instructions
        cv2.putText(
            frame,
            "N: Set Name | E: Enroll | Q: Quit",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow("Face Recognition", frame)

        key = cv2.waitKey(1) & 0xFF

        # Set the current enrollment name
        if key == ord("n"):
            name = input("Enter person's name: ").strip().lower()

            if name:
                enrollment_name = name
                print(f"Enrollment name set to: {enrollment_name}")
            else:
                print("Name cannot be empty!")

        # Save an embedding for the selected person
        elif key == ord("e"):
            if enrollment_name is None:
                print("Press N to enter a name first!")

            elif len(embeddings) == 1:
                db.add_embedding(enrollment_name, embeddings[0])
                known_faces.append(
                    (enrollment_name, embeddings[0])
                )

                print(f"Embedding saved for {enrollment_name}!")

            else:
                print("Enrollment requires exactly one face!")

        elif key == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    db.close()