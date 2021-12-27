import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode = False, maxHands = 2, modelComplexity = 1, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.modelComplexity = modelComplexity

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        
    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo = 0, draw = True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)
        return lmlist
    
    def findTwoHandPositions(self, img, draw = True):
        lmlists = {'RIGHT': [], 'LEFT': []}
        # print("multi_handedness: ", self.results.multi_handedness)
        for hand_idx, hand_info in enumerate(self.results.multi_handedness):
            hand_label = hand_info.classification[0].label
            oneHand = self.results.multi_hand_landmarks[hand_idx]
            for id, lm in enumerate(oneHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlists[hand_label.upper()].append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)
        return lmlists

    
    def fingersUp(self, img):
        count = {'RIGHT': 0, 'LEFT': 0}

        fingers_tips_ids = [self.mpHands.HandLandmark.INDEX_FINGER_TIP, self.mpHands.HandLandmark.MIDDLE_FINGER_TIP, self.mpHands.HandLandmark.RING_FINGER_TIP, self.mpHands.HandLandmark.PINKY_TIP]

        finger_statuses = {'RIGHT_THUMB': False, 'RIGHT_INDEX': False, 'RIGHT_MIDDLE': False, 'RIGHT_RING': False, 'RIGHT_PINKY': False,
                        'LEFT_THUMB': False, 'LEFT_INDEX': False, 'LEFT_MIDDLE': False, 'LEFT_RING': False, 'LEFT_PINKY': False}
        
        for hand_index, hand_info in enumerate(self.results.multi_handedness):
        # print("hand_info: ", hand_info)
            hand_label = hand_info.classification[0].label
            hand_landmarks = self.results.multi_hand_landmarks[hand_index]
            for tip_index in fingers_tips_ids:
                finger_name = tip_index.name.split("_")[0]
                if (hand_landmarks.landmark[tip_index].y < hand_landmarks.landmark[tip_index - 2].y):
                    finger_statuses[hand_label.upper()+"_"+finger_name] = True
                    count[hand_label.upper()] += 1

            thumb_tip_x = hand_landmarks.landmark[self.mpHands.HandLandmark.THUMB_TIP].x
            thumb_mcp_x = hand_landmarks.landmark[self.mpHands.HandLandmark.THUMB_TIP - 2].x

            if (hand_label == 'Right' and (thumb_tip_x < thumb_mcp_x)) or (hand_label == 'Left' and (thumb_tip_x > thumb_mcp_x)):
                finger_statuses[hand_label.upper() + "_THUMB"] = True
                count[hand_label.upper()] += 1
        
        return count, finger_statuses


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img_flipped = cap.read()
        img = cv2.flip(img_flipped, 1)
        img = detector.findHands(img)
        lmlist = detector.findPosition(img)
        if len(lmlist) != 0:
            print(lmlist[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break


if __name__ == "__main__":
    main()