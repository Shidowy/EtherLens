import sys
import cv2
import math
from detection.HandDetection import HandDetector

# Function to calculate and display the distance between index finger and thumb
def main():
    obj_x, obj_y = 200, 200
    obj_width, obj_height = 200, 120  
    buffer_zone = 30  
    dragging = False  
    x, y = 30, 30
    min_size, max_size = 50, 300  # Set limits for the rectangle's size

    hand_detector = HandDetector()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame, results = hand_detector.detect_hands(frame)

        if results.multi_hand_landmarks:
            # Get the index finger tip and thumb tip coordinates
            hand_landmarks = results.multi_hand_landmarks[0]
            index_finger_tip = hand_landmarks.landmark[8]  # Index finger tip
            thumb_tip = hand_landmarks.landmark[4]  # Thumb tip
            middle_finger_tip = hand_landmarks.landmark[12]
            
            # Convert normalized coordinates to pixel values
            index_x, index_y = int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0])
            thumb_x, thumb_y = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])
            x, y = int(middle_finger_tip.x * frame.shape[1]), int(middle_finger_tip.y * frame.shape[0])

            # Calculate the distance between the index finger and thumb tips
            distance = int(math.hypot(index_x - thumb_x, index_y - thumb_y))

            # Display the distance on the screen
            cv2.putText(frame, f"Distance: {distance}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw a line between the index finger and thumb tips
            cv2.line(frame, (index_x, index_y), (thumb_x, thumb_y), (0, 255, 0), 2)

            # Resize the rectangle based on the distance, within a defined range
            obj_width = max(min_size, min(max_size, distance))
            obj_height = max(min_size, min(max_size, distance * 0.6))  # Adjust height based on width

        # Check if middle finger is in the draggable area of the rectangle
        if (obj_x - buffer_zone <= x <= obj_x + obj_width + buffer_zone and 
            obj_y - buffer_zone <= y <= obj_y + obj_height + buffer_zone):
            dragging = True
        else:
            dragging = False
        if dragging:
            obj_x, obj_y = x - obj_width // 2, y - obj_height // 2

        # Draw the draggable rectangle with integer coordinates
        cv2.rectangle(frame, (int(obj_x), int(obj_y)), (int(obj_x + obj_width), int(obj_y + obj_height)), (255, 0, 0), -1)

        # Display the frame
        cv2.imshow("Hand Tracking Distance Measurement", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
