###FACE DETECTION

import cv2
import os

detection_done=False
if detection_done:
    print("face detection already done")
    
# Define the path to the Haar Cascade classifier file
cascade_path = 'haarcascade_frontalface_default.xml'

# Check if the Haar Cascade file exists
if not os.path.isfile(cascade_path):
    print(f"Error: The file {cascade_path} was not found.")
    exit()

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print("Error: Could not load Haar Cascade classifier.")
    exit()

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 is the default ID for the primary webcam

if not cap.isOpened():
    print("Error: Could not open video stream from webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    num_faces=len(faces)
    print("number of faces detected:",{num_faces})

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.putText(frame, f'Faces detected: {num_faces}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the windows
cap.release()
cv2.destroyAllWindows()
detection_done=True


### OBJECT DETECTION
import cv2
import numpy as np
import os

# Download necessary files if not already downloaded
if not os.path.isfile('yolov3.cfg'):
    os.system('wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg')

if not os.path.isfile('yolov3.weights'):
    os.system('wget https://pjreddie.com/media/files/yolov3.weights')

if not os.path.isfile('coco.names'):
    os.system('wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names')

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 is the default ID for the primary webcam

if not cap.isOpened():
    print("Error: Could not open video stream from webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    height, width, channels = frame.shape

    # Prepare the frame to run through the deep neural network
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Display the information on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Filter out weak detections
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the resulting frame with detected objects
    cv2.imshow('Object Detection', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the windows
cap.release()
cv2.destroyAllWindows()
