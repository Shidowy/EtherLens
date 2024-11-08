import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self):
        self.mp_hand = mp.solutions.hands
        self.hand_detector = self.mp_hand.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils
        self.color = (0, 255, 0) 
        self.line_color = (0, 0, 255) 

    def detect_hands(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = self.hand_detector.process(rgb_frame)

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hand.HAND_CONNECTIONS)
                for i, landmark in enumerate(hand_landmarks.landmark):
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, self.color, -1)

                thumb_tip = hand_landmarks.landmark[4]  
                index_tip = hand_landmarks.landmark[8]  

                
                thumb_coords = (int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0]))
                index_coords = (int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0]))

                # draw line
                cv2.line(frame, thumb_coords, index_coords, self.line_color, 2)

        return frame, hand_results

    def release(self):
        self.hand_detector.close()
