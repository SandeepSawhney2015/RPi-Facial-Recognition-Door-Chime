
# Raspberry Pi Smart Doorbell

An AI-powered smart doorbell that uses computer vision to detect and eventually recognize people entering an apartment.

This project is my first hands-on experience with pretrained computer vision models. I'm developing and testing the initial prototype on my MacBook using Python, OpenCV, and YOLO, with plans to deploy it on a Raspberry Pi 5 equipped with the Raspberry Pi AI Camera.

## Project Goals

- **Object Detection:** Detect people and everyday objects in real time using a pretrained YOLO model.
- **Facial Recognition:** Identify familiar visitors using facial embeddings and a local database.
- **Personalized Greetings:** Play a customized chime or greeting when a registered person enters.
- **Entrance Monitoring:** Detect activity around the apartment entrance.
- **Notifications:** Send entrance events and recognition results to authorized users.

## Tech Stack

- **Language:** Python
- **Computer Vision:** OpenCV
- **Object Detection:** Ultralytics YOLO11 Nano
- **Hardware (Planned):** Raspberry Pi 5 + Raspberry Pi AI Camera
- **Database (Planned):** SQLite for storing registered identities and recognition data

## Current Progress

- [x] Configure Python development environment
- [x] Implement live webcam capture with OpenCV
- [x] Integrate pretrained YOLO object detection
- [x] Display real-time bounding boxes and confidence scores
- [ ] Implement facial detection and recognition
- [ ] Store registered facial embeddings locally
- [ ] Integrate Raspberry Pi AI Camera
- [ ] Implement personalized audio greetings
- [ ] Add entrance event notifications

## Development

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the prototype:

```bash
python main.py
```

Press `q` to close the camera feed.

## Project Status

**In Development** — Currently prototyping object detection on macOS before transitioning to Raspberry Pi hardware.

## Privacy

The project is intended for a private apartment entrance. Facial recognition will be limited to enrolled, consenting individuals, with local storage of recognition data and appropriate access controls.

