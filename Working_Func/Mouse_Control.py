import cv2  # Importing the OpenCV library for computer vision
import numpy as np  # Importing NumPy for numerical operations
import HandTrackingModule as htm  # Importing a custom hand tracking module
import time  # Importing the time module for time-related operations
import  pyautogui # Importing pyautogui for mouse and keyboard control

# ------------------- CONFIGURATIONS -------------------

# Camera and Screen Configurations
wCam, hCam = 400, 300  # Setting up the camera width and height
wScr, hScr = pyautogui.size()  # Getting the screen size using pyautogui

# Frame Configurations
frameR = 100  # Frame Reduction (used for creating a frame around the hand)

# Mouse Movement Configurations
smoothening = 4  # Smoothening factor for mouse movement (controls how smoothly the cursor moves)
move_distance_threshold = 48  # Minimum distance to trigger mouse movement
durationCloseThreshold = 0.25  # Minimum duration to hold for a click

# Click Configurations
click_distance_threshold = 39  # Minimum distance to trigger a click
left_click_landmarks = (8, 4)  # Landmarks for left click (referring to finger landmarks)
right_click_landmarks = (12, 4)  # Landmarks for right click (referring to finger landmarks)

# Scroll Configurations
scroll_speed = 5  # Speed of scrolling
scroll_threshold = 30  # Minimum distance to trigger scrolling

# ------------------- INITIALIZATIONS -------------------

# Initializing the hand detector module from the custom module "HandTrackingModule"
detector = htm.handDetector(maxHands=1)

# Disabling the failsafe for pyautogui (allows mouse to move to screen edges)
pyautogui.FAILSAFE = False

# Variables to keep track of previous and current mouse coordinates
plocX, plocY = 0, 0
clocX, clocY = 0, 0

# Variables to handle click actions
heldStartTime = None
isHeldClick = False
isLeftClick = False
isRightClick = False

# Variable to keep track of the previous time
pTime = 0


# Function to move the mouse cursor
def move_mouse(x1, y1):
    # Interpolate hand position to screen coordinates
    x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
    y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

    # Smoothly move the cursor to the new position
    clocX = plocX + (x3 - plocX) / smoothening
    clocY = plocY + (y3 - plocY) / smoothening

    # Move the mouse cursor to the calculated position using pyautogui
    pyautogui.moveTo(wScr - clocX, clocY)

    return clocX, clocY


# Function to handle mouse clicks
def handle_mouse_clicks(length, lineInfo, click_type):
    global isHeldClick, isLeftClick, isRightClick, heldStartTime

    # Check if landmarks are close enough for a click action
    if length < click_distance_threshold:
        if heldStartTime is None:
            heldStartTime = time.time()
        durationClose = time.time() - heldStartTime
        if durationClose > durationCloseThreshold and not isHeldClick:
            pyautogui.mouseDown()
            isHeldClick = True
        elif durationClose < durationCloseThreshold and not isHeldClick:
            if click_type == 'left' and not isLeftClick:
                pyautogui.click(button='left')
                isLeftClick = True
            elif click_type == 'right' and not isRightClick:
                pyautogui.click(button='right')
                isRightClick = True
    else:
        heldStartTime = None
        if isHeldClick and length > (click_distance_threshold * 1.5):  # Adjust multiplier as needed
            pyautogui.mouseUp()
            isHeldClick = False
        isLeftClick = False
        isRightClick = False


# Function to handle scrolling
def handle_scrolling(y1, y2, fingers):
    scroll_distance = abs(y1 - y2)
    if scroll_distance > scroll_threshold:
        scroll_amount = scroll_speed * (scroll_distance - scroll_threshold)
        if all(fingers[i] == 1 for i in range(1, 5)):  # Checking if all fingers except the thumb are up
            pyautogui.scroll(scroll_amount)  # Scroll up
        elif all(fingers[i] == 0 for i in range(2, 5)):  # Checking if all fingers except the thumb are down
            pyautogui.scroll(-scroll_amount)  # Scroll down


# Initialize the camera
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# ------------------- MAIN LOOP -------------------
# In this code, we continuously capture frames from the camera, process hand landmarks,
# and update the mouse cursor, clicks, and scrolls based on hand gestures.

while True:
    # Read a frame from the camera
    success, img = cap.read()

    # Find hands in the frame using the custom hand tracking module
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    if len(lmList) != 0:
        # List of hand landmarks:
        # In a hand, there are 21 landmarks represented by indexes from 0 to 20.
        # These landmarks represent various points on the hand, such as fingertips and joints.
        # More details about each landmark can be found in the Mediapipe documentation:
        # https://google.github.io/mediapipe/solutions/hands#hand-landmarks
        # Extract hand landmarks and display them on the image
        lm5 = lmList[5][1:]  # Extracting the position of landmark 5 (a finger)
        lm8 = lmList[8][1:]  # Extracting the position of landmark 8 (a finger)
        lm12 = lmList[12][1:]  # Extracting the position of landmark 12 (a finger)
        cv2.putText(img, f'5: {lm5}', (lm5[0] + 10, lm5[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        cv2.putText(img, f'8: {lm8}', (lm8[0] + 10, lm8[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        cv2.putText(img, f'12: {lm12}', (lm12[0] + 10, lm12[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

        # Display rectangle corners for reference
        rect_coords = [(frameR, frameR), (wCam - frameR, frameR), (wCam - frameR, hCam - frameR),
                       (frameR, hCam - frameR)]
        for coord in rect_coords:
            cv2.putText(img, str(coord), (coord[0] + 10, coord[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

        # Get hand positions and finger states
        x1, y1 = lmList[5][1:]
        x2, y2 = lmList[4][1:]
        fingers = detector.fingersUp()

        # Create a rectangle on the image for visual reference Parameters: - img: The image on which the rectangle
        # will be drawn. - (frameR, frameR): The top-left corner coordinates of the rectangle. These represent the (
        # x, y) coordinates. - `frameR` is used to create an inset frame around the image. - (wCam - frameR,
        # hCam - frameR): The bottom-right corner coordinates of the rectangle. - These coordinates are calculated
        # based on the width and height of the camera frame minus the inset frame size. - (255, 0, 255): The color of
        # the rectangle in BGR format. Here, it's magenta. - 1: The thickness of the rectangle's border.
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 1)

        # Handle mouse movement if index and middle fingers are extended
        if fingers[1] == 1 and fingers[0] == 1:
            length, img, lineInfo = detector.findDistance(8, 4, img)
            if length > move_distance_threshold:
                plocX, plocY = move_mouse(x1, y1)

        # Handle left and right clicks
        length, img, lineInfo = detector.findDistance(*left_click_landmarks, img)
        handle_mouse_clicks(length, lineInfo, 'left')
        length, img, lineInfo = detector.findDistance(*right_click_landmarks, img)
        handle_mouse_clicks(length, lineInfo, 'right')

        # Handle scrolling
        handle_scrolling(y1, y2, fingers)

    # Calculate and display frames per second (FPS)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # Show the processed image with annotations
    cv2.imshow("Image", img)

    # Wait for a key press and check if it's ESC
    if cv2.waitKey(1) == 27:  # 27 is the ASCII value of ESC
        break
