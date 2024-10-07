from ultralytics import YOLO
import cv2

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # Update the path if necessary

# Open video file
video_path = r"C:\projectcomputergraphics\source\455409_Brussels_Bruxelles_1280x720.mp4"
cap = cv2.VideoCapture(video_path)

# Define a dictionary to hold color mapping for classes
color_mapping = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference
    results = model(frame)

    # Loop over each detected object
    for result in results:
        # Extract data for each detection
        for detect in result.boxes:
            class_id = int(detect.cls)
            confidence = float(detect.conf)

            # Extract and format bounding box coordinates
            if detect.xyxy.shape[0] == 1:  # Ensure there's one detection
                x1, y1, x2, y2 = map(int, detect.xyxy[0].tolist())
            else:
                continue  # or handle differently

            # Get class name
            class_name = result.names[class_id]

            # Assign a unique color to each class if not already assigned
            if class_name not in color_mapping:
                # Random color addition for each class
                color_mapping[class_name] = (int(class_id * 37 % 256), int(class_id * 57 % 256), int(class_id * 97 % 256))

            color = color_mapping[class_name]

            # Draw rectangle and add text
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{class_name}: {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display result
    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Break on 'q' key press
        break

# Release resources
cap.release()
cv2.destroyAllWindows()




