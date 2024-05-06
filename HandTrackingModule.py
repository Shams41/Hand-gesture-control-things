import cv2
import mediapipe as mp
import time
import math
import numpy as np

# Creating a class named handDetector to encapsulate all hand detection functionality.
class handDetector():
    # Initializing the handDetector object with various parameters for hand detection.
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        # Setting up the parameters for the hand detection model.
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # Initializing the MediaPipe Hands model.
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity,
                                        self.detectionCon, self.trackCon)
        # For drawing hand landmarks and connections.
        self.mpDraw = mp.solutions.drawing_utils
        # Identifying the tip of the fingers in the landmarks list.
        self.tipIds = [4, 8, 12, 16, 20]

    # Method to find hands in the image and optionally draw the landmarks and connections.
    def findHands(self, img, draw=True):
        # Converting the image to RGB as the model expects RGB images.
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Processing the image to find hand landmarks.
        self.results = self.hands.process(imgRGB)

        # If landmarks are found, draw them.
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # Drawing landmarks and connections.
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    # Method to find the position of landmarks and optionally draw them.
    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        # If landmarks are found, process them.
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                # Calculating the x and y coordinates of the landmarks.
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    # Drawing a circle at each landmark.
                    cv2.circle(img, (cx, cy), 1, (255, 0, 255), cv2.FILLED)

            # Getting the bounding box coordinates.
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                # Drawing a bounding box around the hand.
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)

        return self.lmList, bbox

    # Method to check which fingers are open.
    def fingersUp(self):
        fingers = []
        # Checking thumb.
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Checking other fingers.
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    # Method to find the distance between two points and optionally draw the line and points.
    def findDistance(self, p1, p2, img, draw=True,r=3, t=2):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        # Calculating the Euclidean distance between the two points.
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]

# Main function to run the hand detection and draw results.
def main():
    pTime = 0
    cTime = 0
    # Starting the webcam feed.
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        # Calculating frames per second (FPS).
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # Displaying the FPS on the window.
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        # Displaying the image window.
        cv2.imshow("Image", img)
        cv2.waitKey(1)

# Running the main function when the script is executed.
if __name__ == "__main__":
    main()
