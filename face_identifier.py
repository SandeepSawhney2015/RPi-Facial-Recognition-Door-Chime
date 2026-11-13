import cv2

class FaceIdentifier:
    def __init__(self):
        self.detector = cv2.FaceDetectorYN.create("models/yunet.onnx", "", (320, 320))
        self.recognizer = cv2.FaceRecognizerSF.create("models/sface.onnx", "")

    def get_embeddings(self, frame):
        height, width = frame.shape[:2] #getting the dimensions from the cropping
        #(height, width, color_channels)
        self.detector.setInputSize((width, height))

        _, faces = self.detector.detect(frame) #gets information from the frame

        if faces is None:
            return []

        embeddings = []

        for face in faces:
            aligned = self.recognizer.alignCrop(frame, face)
            embedding = self.recognizer.feature(aligned)
            embeddings.append(embedding)

        return embeddings
