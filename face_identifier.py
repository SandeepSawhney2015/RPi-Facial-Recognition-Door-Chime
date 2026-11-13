
import cv2

class FaceIdentifier:
    def __init__(self):
        self.detector = cv2.FaceDetectorYN.create(
            "models/yunet.onnx", "", (320, 320)
        )

        self.recognizer = cv2.FaceRecognizerSF.create(
            "models/sface.onnx", ""
        )

    def get_face_data(self, frame):
        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))

        _, faces = self.detector.detect(frame)

        if faces is None:
            return []

        results = []

        for face in faces:
            # YuNet's first 4 values are x, y, width, height
            x, y, w, h = map(int, face[:4])

            # Generate embedding
            aligned = self.recognizer.alignCrop(frame, face)
            embedding = self.recognizer.feature(aligned)

            # Store both the embedding and face location
            results.append((embedding, (x, y, w, h)))

        return results

    # Keep this for your existing test_faces.py
    def get_embeddings(self, frame):
        return [
            embedding
            for embedding, box in self.get_face_data(frame)
        ]
