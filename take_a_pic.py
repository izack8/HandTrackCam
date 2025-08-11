import cv2
import os
cap = cv2.VideoCapture(0)

image_counter = 0

file_path = "./data/none/"
os.makedirs(file_path, exist_ok=True)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    frame = cv2.flip(frame, 1)

    cv2.imshow("Capture Image", frame)


    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):  
        image_name = f"{image_counter}.png"
        cv2.imwrite(file_path + image_name, frame)
        print(f"Image saved as {image_name}")
        image_counter += 1

    elif key == ord('q'):  
        print("Exiting...")
        break


cap.release()
cv2.destroyAllWindows()