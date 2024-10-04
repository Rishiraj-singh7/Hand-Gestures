import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize the hand detector
detector = HandDetector(maxHands=2, detectionCon=0.5, minTrackCon=0.5)

while True:
    # Read a frame from the webcam
    success, im = cap.read()
    if not success:
        break

    # Flip the image horizontally for a mirror effect
    im = cv2.flip(im, 1)

    # Detect hands in the image
    hands, im = detector.findHands(im, draw=True)
    
    # Initialize counts
    count_1 = 0
    count_0 = 0
    
    # If two hands are detected
    if len(hands) == 2:
        hand = hands[0]
        hand1 = hands[1]
        
        # Get the landmark list for each hand
        lmlist = hand['lmList']
        lmlist1 = hand1['lmList']
        
        # Get the coordinates of the tip of the thumb for each hand
        cx = lmlist[4][0]
        cy = lmlist[4][1]
        cx1 = lmlist1[4][0]
        cy1 = lmlist1[4][1]
        
        # Uncomment the lines below to draw circles on the thumb tips
        # cv2.circle(im, (cx, cy), 13, (255, 0, 0), -1)
        # cv2.circle(im, (cx1, cy1), 13, (255, 255, 0), -1)

        # Get the status of fingers (up or down) for each hand
        fingers1 = detector.fingersUp(hand)
        fingers2 = detector.fingersUp(hand1)
        
        # Combine the status lists of both hands
        list = fingers1 + fingers2
        
        # Count the number of fingers that are up and down
        count_1 = list.count(1)
        count_0 = list.count(0)

    # Draw background rectangles for the text
    cv2.rectangle(im, (10, 20), (300, 70), (0, 0, 0), -1)  # Background for "Count of 1s"
    cv2.rectangle(im, (10, 70), (300, 120), (0, 0, 0), -1)  # Background for "Count of 0s"

    # Display the counts on the image
    cv2.putText(im, f"Count of 1s: {count_1}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(im, f"Count of 0s: {count_0}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the image
    cv2.imshow("im", im)
    
    # Break the loop if 'esc' key is pressed
    key = cv2.waitKey(1)
    if key == 27:  # esc
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
