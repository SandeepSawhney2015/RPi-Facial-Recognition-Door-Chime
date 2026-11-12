import cv2
from ultralytics import YOLO
import sys

camera = cv2.VideoCapture(0) #this opens the default macbook camera I think

if not camera.isOpened() :
    print("could not open camera, make sure you allow the system to open your camera (there should be a pop up!)", file=sys.stderr)
    sys.exit(1)

model = YOLO("yolo11n.pt") #fast and lightweight model 

#success is a boolean statement and is true if a frame was captured
'''
for i in range(30):
    success, frame = camera.read()

if success :
    cv2.imshow("Camera detection", frame)
    cv2.waitKey(0)
'''

while True:
    success, frame = camera.read()

    if not success:
        print("failed to read the frame", sys.stderr)
        sys.exit(1)

    results = model(frame, verbose=False)

    annotatedFrame = results[0].plot()

    cv2.imshow("Camera detection", annotatedFrame)

    if cv2.waitKey(1) & 0xFF == ord('q'): #this I believes waits 1ms for a q keystroke
        break
        

camera.release
cv2.destroyAllWindows