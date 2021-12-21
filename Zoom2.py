import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.7)


def recognizeGesture(hands):
    hand_gestures = {"RIGHT": "Unknown", "LEFT": "Unknown"}
    gestures = "Unknown"

    # identify the gestures of two hands
    for hand_idx, hand_label in enumerate(hand_gestures):
        # print(hand_idx, hand_label)
        if detector.fingersUp(hands[hand_idx]) == [1,1,0,0,0]:
            hand_gestures[hand_label] = "V Sign"
        elif detector.fingersUp(hands[hand_idx]) == [0,1,1,0,0]:
            hand_gestures[hand_label] = "Yea"

    # see what type of gesture
    if hand_gestures["RIGHT"] == "V Sign" and hand_gestures["LEFT"] == "V Sign":
        gestures = "Zoom"
    elif hand_gestures["RIGHT"] == "Yea":
        gestures = "Scroll"

    return hand_gestures, gestures

def calculateZoom(hands):
    lmlist0 = hands[0]["lmList"]
    lmlist1 = hands[1]["lmList"]

    length, info = detector.findDistance(lmlist0[8], lmlist1[8])

    return length, info

def main():
    
    startDist_z = None
    cx, cy = 500, 500

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        hands, img = detector.findHands(img)
        # hands, img = detector.findHands(img)
        image1 = cv2.imread('smallotter.png')

        # print(hands)

        if len(hands) == 2:
            hand_gestures, gestures = recognizeGesture(hands)
            print(hand_gestures, gestures)
            if gestures == "Zoom":
                length, info = calculateZoom(hands)
                if startDist_z is None: 
                    startDist_z = length

                scale = int((length - startDist_z) // 2)
                print("SCALE: ", scale)
                cx, cy = info[4:]
                h1, w1, _ = image1.shape
                newH, newW = h1 + scale, w1 + scale
                image1 = cv2.resize(image1, (newH - newH % 2, newW - newW % 2))
                print(image1.shape)
                print((cy+newH//2)-(cy-newH//2),(cx+newW//2)-(cx-newW//2))
                img[cy-newH//2 : cy+newH//2, cx-newW//2 : cx+newW//2] = image1

            else: 
                startDist_z = None

        
        cv2.imshow("Image", img)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

if __name__ == "__main__":
    main()