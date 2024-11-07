import cv2
from detection.HandArmDetection import HandDetector
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

def main():
    detector = HandDetector()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        # Detect hands
        frame, hand_results = detector.detect_hands(frame)

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                for i, landmark in enumerate(hand_landmarks.landmark):
                    print(f"Hand Landmark {i}: x={landmark.x}, y={landmark.y}, z={landmark.z}")

        cv2.imshow("Hand Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
