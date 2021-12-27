import cv2
import HandTrackingModule as htm
import numpy as np
import autopy

wCam, hCam = 640, 480
wScr, hScr = autopy.screen.size() # 1440, 900

def findHandGesture(count, finger_statuses, lmlists):
    hand_gestures = {'RIGHT': "Unknown", 'LEFT': "Unknown"}

    for hand_index, hand_label in enumerate(hand_gestures):
        
        # see if this hand is present
        if hand_gestures[hand_label] == []:
            hand_gestures[hand_label] = "No hand"
            continue
        
        # check gesture of that hand
        if count[hand_label] == 2:
            if finger_statuses[hand_label + '_THUMB'] and finger_statuses[hand_label + '_INDEX']:
                dist35 = abs(lmlists[hand_label][3][1] - lmlists[hand_label][5][1])
                # print("dist35: ", dist35)
                if dist35 < 80:
                    hand_gestures[hand_label] = "Snap"
                else:
                    hand_gestures[hand_label] = "V Sign"

            if finger_statuses[hand_label + '_INDEX'] and finger_statuses[hand_label + '_MIDDLE']:
                hand_gestures[hand_label] = "Yea"
        
        elif count[hand_label.upper()] == 1:
            if finger_statuses[hand_label + '_INDEX']:
                hand_gestures[hand_label] = "One"

    return hand_gestures

def recoMode(hand_gestures):
    mode = "Unknown"

    if hand_gestures['RIGHT'] == "V Sign" and hand_gestures['LEFT'] == "V Sign":
        mode = "Zoom"
    elif hand_gestures['RIGHT'] == "Yea":
        mode = "Scroll"
    elif hand_gestures['RIGHT'] == "One":
        mode = "Cursor"
    elif hand_gestures['RIGHT'] == "Snap":
        mode = "Click"

    return mode

def moveCursor(x1, y1):
    x3 = np.interp(x1, (0, wCam), (0, wScr))
    y3 = np.interp(y1, (9, hCam), (0, hScr))

    autopy.mouse.move(x3, y3)


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)
    detector = htm.handDetector(detectionCon=0.7)

    image1 = cv2.imread('smallotter.png')

    while True:
        successs, img = cap.read()
        frame = cv2.flip(img, 1)
        frame = detector.findHands(frame)
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.hands.process(imgRGB)
        
        if results.multi_hand_landmarks:
            count, finger_statuses = detector.fingersUp(frame)
            lmlists = detector.findTwoHandPositions(frame, draw=False)
            # print(lmlists)
            # print(len(lmlists['RIGHT']), len(lmlists['LEFT']))
            
            hand_gestures = findHandGesture(count, finger_statuses, lmlists)
            mode = recoMode(hand_gestures)
            if mode == "Cursor":
                x1 = lmlists['RIGHT'][8][1]
                y1 = lmlists["RIGHT"][8][2]
                cv2.circle(frame, (x1, y1), 7, (127, 255, 0), cv2.FILLED)
                moveCursor(x1, y1)

        cv2.imshow('Virtual Mouse', frame)

        k = cv2.waitKey(1)
        if k == ord('q'):
            break

if __name__ == "__main__":
    main()