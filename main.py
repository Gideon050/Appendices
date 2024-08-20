import cv2
from picamera2 import Picamera2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
import threading
import time
from gpiozero import LED, Buzzer
from time import sleep

# Initialize GPIO devices for LED indicators and buzzer
GREEN_LED = LED(17)   # Green LED indicates the door is open
RED_LED = LED(27)     # Red LED indicates the door is closed
YELLOW_LED = LED(22)  # Yellow LED indicates detection in progress
BLUE_LED = LED(23)    # Blue LED indicates system in standby mode
BUZZER = Buzzer(24)   # Buzzer sounds during door action

# Initialize Picamera2 for capturing video
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)  # Set resolution for faster processing
picam2.preview_configuration.main.format = "RGB888"  # Set format for image capture
picam2.preview_configuration.align()                 # Align the configuration
picam2.configure("preview")                          # Configure the camera for preview mode
picam2.start()                                       # Start the camera

# Load the YOLO model for object detection and the class list
model = YOLO('best.pt')  # Load the pre-trained YOLO model
with open("coco2.txt", "r") as f:
    class_list = f.read().split("\n")  # Load the class names from the file

# Variables to track previous bounding box positions for movement direction calculation
prev_bbox = None
prev_centroid = None
frame_lock = threading.Lock()  # Lock to synchronize frame access
frame = None  # Variable to store the captured frame
door_state = "closed"  # Track the current state of the door

# Function to calculate the direction of movement based on bounding box coordinates
def calculate_direction(bbox):
    global prev_bbox, prev_centroid

    # Calculate the centroid of the current bounding box
    current_centroid = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

    # If there is no previous centroid, initialize it and return "approaching"
    if prev_centroid is None:
        prev_centroid = current_centroid
        prev_bbox = bbox
        return "approaching"

    # Calculate the difference in position
    dx = current_centroid[0] - prev_centroid[0]
    dy = current_centroid[1] - prev_centroid[1]

    # Update previous centroid and bounding box
    prev_centroid = current_centroid
    prev_bbox = bbox

    # Determine direction based on movement
    if dx > 0 and dy > 0:
        return "approaching"
    else:
        return "moving away"

# Function to continuously capture frames from the camera
def capture_frames():
    global frame
    while True:
        with frame_lock:
            frame = picam2.capture_array()  # Capture frame as an array
        time.sleep(0.03)  # Wait to maintain approximately 30 FPS

# Start a separate thread for capturing frames
capture_thread = threading.Thread(target=capture_frames)
capture_thread.daemon = True  # Daemon thread to stop with the main program
capture_thread.start()

# Function to control the door based on the command
def control_door(command):
    global door_state
    if command == "open" and door_state != "open":
        print("Open the door")
        GREEN_LED.on()   # Turn on the green LED
        RED_LED.off()    # Turn off the red LED
        BUZZER.on()      # Sound the buzzer
        time.sleep(5)    # Buzzer duration in seconds
        BUZZER.off()     # Stop the buzzer
        door_state = "open"
    elif command == "close" and door_state != "close":
        print("Close the door")
        GREEN_LED.off()  # Turn off the green LED
        RED_LED.on()     # Turn on the red LED
        BUZZER.off()     # Ensure the buzzer is off
        door_state = "close"

# Function to control the detection LED
def control_detection_led(action):
    if action == "detect":
        YELLOW_LED.on()  # Turn on the yellow LED during detection
    else:
        YELLOW_LED.off()  # Turn off the yellow LED

# Function to control the standby LED
def control_standby_led(action):
    if action == "standby":
        BLUE_LED.on()  # Turn on the blue LED in standby mode
    else:
        BLUE_LED.off()  # Turn off the blue LED

# Initialize a counter for processing frames
count = 0

# Main loop for processing the captured frames
try:
    while True:
        if frame is None:
            continue  # Skip if no frame is captured
        
        with frame_lock:
            im = frame.copy()  # Make a copy of the current frame
        
        count += 1
        if count % 5 != 0:
            continue  # Process every 5th frame to reduce load
        
        im = cv2.flip(im, 1)  # Flip the image horizontally for mirror view

        try:
            results = model.predict(im)  # Perform object detection on the frame
        except Exception as e:
            print(f"Model prediction error: {e}")
            continue  # Skip to the next frame if prediction fails
        
        person_detected = False  # Flag to check if a person is detected
        
        if results and len(results[0].boxes) > 0:
            a = results[0].boxes.data.cpu().numpy()  # Convert detection results to numpy array
            px = pd.DataFrame(a).astype("float")  # Convert to DataFrame for easier processing
        
            for index, row in px.iterrows():
                x1, y1, x2, y2, _, d = map(int, row)  # Extract bounding box coordinates and class id
                c = class_list[d]  # Get the class name based on the class id
                
                bbox = [x1, y1, x2, y2]  # Define the bounding box
                direction = calculate_direction(bbox)  # Calculate movement direction
                
                person_detected = True  # Set flag to true if a person is detected

                # Determine door control based on detection
                if c == 'front' and direction == "approaching":
                    control_door("open")  # Open the door if someone is approaching from the front
                    control_detection_led("detect")  # Turn on detection LED
                elif c == 'side':
                    control_door("close")  # Close the door if detected from the side
                    control_detection_led("detect")  # Turn on detection LED
                
                # Draw bounding box and label on the image
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cvzone.putTextRect(im, f'{c}', (x1, y1), 1, 1)
        
        if not person_detected:
            control_detection_led("off")  # Turn off detection LED if no person is detected
            control_door("close")  # Ensure the door is closed
            control_standby_led("standby")  # Turn on standby LED
        else:
            control_standby_led("off")  # Turn off standby LED if a person is detected
        
        # Display the processed image
        cv2.imshow("Camera", im)
        if cv2.waitKey(1) == ord('q'):
            break  # Exit the loop if 'q' is pressed

# Handle cleanup when the program is interrupted
except KeyboardInterrupt:
    pass
finally:
    cv2.destroyAllWindows()  # Close all OpenCV windows
    picam2.stop()  # Stop the camera
    control_door("close")  # Ensure the door is closed
    control_detection_led("off")  # Turn off all LEDs
    control_standby_led("off")
    GREEN_LED.off()
    RED_LED.off()
    YELLOW_LED.off()
    BLUE_LED.off()
    BUZZER.off()  # Ensure the buzzer is off
