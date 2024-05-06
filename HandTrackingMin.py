import cv2
import mediapipe as mp
import time

# Start capturing video from the first camera device.
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands model.
mpHands = mp.solutions.hands
hands = mpHands.Hands()
# Initialize MediaPipe Drawing module.
mpDraw = mp.solutions.drawing_utils

# Variables to calculate frames per second (FPS).
pTime = 8
cTime = 0

# Start an infinite loop to continuously get frames from the camera.
while True:
    # Read a frame from the camera.
    success, img = cap.read()

    # Convert the BGR image to RGB since MediaPipe Hands model expects RGB images.
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Process the image to find hand landmarks.
    results = hands.process(imgRGB)

    # Check if any hand landmarks are found.
    if results.multi_hand_landmarks:
        # Loop through each hand found.
        for handLms in results.multi_hand_landmarks:
            # Loop through each landmark in a hand.
            for id, lm in enumerate(handLms.landmark):
                # Get the width, height, and channels of the image.
                h, w, c, = img.shape
                # Calculate the x and y coordinates of the landmark.
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                # If it's the first landmark, draw a circle.
                if id == 0:
                    cv2.circle(img, (cx, cy), 2, (255, 0, 255), cv2.FILLED)
            # Draw the landmarks and connections between them.
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # Calculate the frames per second (FPS).
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Display the FPS on the image.
    cv2.putText(img, str(int(fps)), (18, 78),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # Display the image.
    cv2.imshow("Image", img)
    # Wait for 1 millisecond between frames.
    cv2.waitKey(1)