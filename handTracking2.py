import cv2
import numpy as np

# Create a window to display the video feed
cv2.namedWindow("Hand Tracking")

# Get the camera object
cap = cv2.VideoCapture(0)

# Define the range of colors to track in HSV color space
lower_color = np.array([0, 70, 70])
upper_color = np.array([50, 255, 255])

# Run the webcam until the user stops it
while True:
    # Read the current frame from the webcam
    ret, frame = cap.read()

    # Convert the frame from BGR to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the frame to get only the colors in the defined range
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Perform morphological operations to remove noise
    morph_kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, morph_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, morph_kernel)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw a bounding box around the largest contour
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(largest_contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the video feed
    cv2.imshow("Hand Tracking", frame)

    # Break the loop if the user presses the "q" key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and destroy the window
cap.release()
cv2.destroyAllWindows()
