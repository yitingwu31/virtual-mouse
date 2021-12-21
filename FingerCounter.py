import cv2
import HandTrackingModule as htm
import matplotlib.pyplot as plt
import math

frameWidth = 1280
frameHeight = 720

class scrollCalculator():
    def __init__(self, xorg = 500, yorg = 500, threshold = 20, active=False):
        self.xorg = xorg
        self.yorg = yorg
        self.threshold = threshold
        self.active = active
    
    def scrolling(self, x1, y1):
        if self.active == False:
            self.xorg = x1
            self.yorg = y1
            self.active = True
        if y1 > self.yorg and y1 - self.yorg > self.threshold:
            return "Down"
        elif y1 < self.yorg and self.yorg - y1 > self.threshold:
            return "Up"
        else: 
            return "Same"
    
    def resetScroll(self):
        self.xorg = 500
        self.yorg = 500
        self.active = False

class imageHandler():
    def __init__(self, image1, cx = 500, cy = 500, distance = None, speed = 10):
        self.img = image1
        self.h1, self.w1, _ = image1.shape
        self.h, self.w = self.h1, self.w1
        self.cx = cx
        self.cy = cy
        self.distance = distance
        self.speed = speed

        self.scroller = scrollCalculator()
    
    def zoomImage(self, img, pos0, pos1):
        x1, y1 = pos0[1], pos0[2]
        x2, y2 = pos1[1], pos1[2]
        self.cx, self.cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(abs(x2 - x1), abs(y2 - y1))
        if self.distance is None:
            self.distance = length
        scale = int((length - self.distance) // 2)
        newH, newW = self.h1 + scale, self.w1 + scale
        print("length: ", length, ", scale: ", scale, ", newH: ", newH)

        if (self.cy - newH//2) <= 0 or (self.cy + newH//2) >= frameHeight:
            newH = self.cy * 2
        if (self.cx - newW//2) <= 0 or (self.cx + newH//2) >= frameWidth:
            newW = self.cx * 2

        newimage = cv2.resize(self.img, (newH - newH % 2, newW - newW % 2))
        img[self.cy-newH//2 : self.cy+newH//2, self.cx-newW//2 : self.cx+newW//2] = newimage
        self.h = newH
        self.w = newW
        self.img = newimage
        return img
    
    def scrollImage(self, img, pos):
        x1, y1 = pos[1], pos[2]
        direction = self.scroller.scrolling(x1, y1)
        newcy = self.cy
        if direction == "Up":
            newcy = self.cy - self.speed
        elif direction == "Down":
            newcy = self.cy + self.speed
        img[newcy-self.h//2 : newcy+self.h//2, self.cx-self.w//2 : self.cx+self.w//2] = self.img
        self.cy = newcy
        return img

    def resetImage(self, resetScroll = True, resetSize = False):
        if resetScroll:
            self.scroller.resetScroll()
        if resetSize:
            self.h, self.w = self.h1, self.w1
            self.cx, self.cy = 500, 500
            self.img = cv2.resize(self.img, (self.h1, self.y1))

def countFingers(image, hands, results, draw = True, display = True):
    height, width, _ = image.shape
    output_image = image.copy()
    count = {'RIGHT': 0, 'LEFT': 0}
    hand_gestures = {'RIGHT': "None", 'LEFT': "None"}

    fingers_tips_ids = [hands.HandLandmark.INDEX_FINGER_TIP, hands.HandLandmark.MIDDLE_FINGER_TIP, hands.HandLandmark.RING_FINGER_TIP, hands.HandLandmark.PINKY_TIP]

    finger_statuses = {'RIGHT_THUMB': False, 'RIGHT_INDEX': False, 'RIGHT_MIDDLE': False, 'RIGHT_RING': False, 'RIGHT_PINKY': False,
                        'LEFT_THUMB': False, 'LEFT_INDEX': False, 'LEFT_MIDDLE': False, 'LEFT_RING': False, 'LEFT_PINKY': False}

    # print("results multi: ", results.multi_handedness)
    for hand_index, hand_info in enumerate(results.multi_handedness):
        # print("hand_info: ", hand_info)
        hand_label = hand_info.classification[0].label
        hand_landmarks = results.multi_hand_landmarks[hand_index]
        for tip_index in fingers_tips_ids:
            finger_name = tip_index.name.split("_")[0]
            if (hand_landmarks.landmark[tip_index].y < hand_landmarks.landmark[tip_index - 2].y):
                finger_statuses[hand_label.upper()+"_"+finger_name] = True
                count[hand_label.upper()] += 1

        thumb_tip_x = hand_landmarks.landmark[hands.HandLandmark.THUMB_TIP].x
        thumb_mcp_x = hand_landmarks.landmark[hands.HandLandmark.THUMB_TIP - 2].x

        if (hand_label == 'Right' and (thumb_tip_x < thumb_mcp_x)) or (hand_label == 'Left' and (thumb_tip_x > thumb_mcp_x)):
            finger_statuses[hand_label.upper() + "_THUMB"] = True
            count[hand_label.upper()] += 1

        # see if this hand is present
        if hand_gestures[hand_label.upper()] == "None":
            hand_gestures[hand_label.upper()] = "Unknown"
        
        # check gesture of that hand
        if count[hand_label.upper()] == 2:
            if finger_statuses[hand_label.upper() + '_THUMB'] and finger_statuses[hand_label.upper() + '_INDEX']:
                hand_gestures[hand_label.upper()] = "V Sign"

            if finger_statuses[hand_label.upper() + '_INDEX'] and finger_statuses[hand_label.upper() + '_MIDDLE']:
                hand_gestures[hand_label.upper()] = "Yea"
        
        elif count[hand_label.upper()] == 1:
            if finger_statuses[hand_label.upper() + '_INDEX']:
                hand_gestures[hand_label.upper()] = "One"

        # write out gesture
        cv2.putText(output_image, hand_label + ': ' + hand_gestures[hand_label.upper()], (10, (hand_index+1) * 60), cv2.FONT_HERSHEY_PLAIN, 3, (20, 255, 155), 3)
    
    if draw:
        cv2.putText(output_image, "Total Fingers: ", (10,25), cv2.FONT_HERSHEY_COMPLEX, 1, (20, 255, 155), 2)
        cv2.putText(output_image, str(sum(count.values())), (width//2 - 150, 240), cv2.FONT_HERSHEY_SIMPLEX, 8.9, (20, 255, 155), 10, 10)

    if display:
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1])
        plt.title("Output Image")
        plt.axis('off')
    
    else:
        return output_image, finger_statuses, count, hand_gestures

def recoGesture(hand_gestures):
    mode = "Unknown"
    if hand_gestures['RIGHT'] == "V Sign" and hand_gestures['LEFT'] == "V Sign":
        mode = "Zoom"
    elif hand_gestures['RIGHT'] == "Yea":
        mode = "Scroll"
    elif hand_gestures['RIGHT'] == "One":
        mode = "Cursor"
    
    return mode



def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, frameWidth) # width
    cap.set(4, frameHeight)  # height
    detector = htm.handDetector(detectionCon= 0.7)

    image1 = cv2.imread('./CVZone/smallotter.png')

    imghandler = imageHandler(image1)


    while True:
        success, img_flp = cap.read()
        frame = cv2.flip(img_flp, 1)
        frame = detector.findHands(frame)
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.hands.process(imgRGB)
        if results.multi_hand_landmarks:
            frame, finger_statuses, count, hand_gestures = countFingers(frame, detector.mpHands, results, draw=False, display=False)
            mode = recoGesture(hand_gestures)
            print(mode)
            if mode == "Zoom":
                lmlist0 = detector.findPosition(frame, handNo=0, draw=False)
                lmlist1 = detector.findPosition(frame, handNo=1, draw=False)
                frame = imghandler.zoomImage(frame, lmlist0[8], lmlist1[8])
            elif mode == "Scroll":
                hand_num = 0
                print(results.multi_handedness, results.multi_handedness[0].classification[0].label)
                if results.multi_handedness[0].classification[0].label == "Left":
                    hand_num = 1
                lmlist = detector.findPosition(frame, handNo=hand_num, draw=False)
                frame = imghandler.scrollImage(frame, lmlist[8])
                print("hand_num: ", hand_num)
            elif mode == "Cursor":
                hand_num = 0
                print(results.multi_handedness, results.multi_handedness[0].classification[0].label)
                if results.multi_handedness[0].classification[0].label == "Left":
                    hand_num = 1
                lmlist = detector.findPosition(frame, handNo=hand_num, draw=False)
                _, cx, cy = lmlist[8]
                cv2.circle(frame, (cx, cy), 30, (0, 255, 255), 8)
            else:
                imghandler.resetImage()

        cv2.imshow('Fingers Counter', frame)
        
        k = cv2.waitKey(1)
        if k == ord('q'):
            break


if __name__ == '__main__':
    main()