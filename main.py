import cv2
import numpy as np
import time
import argparse
from ultralytics import YOLO
from collections import deque

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="YOLOv8 Live Webcam Detection")
    parser.add_argument("--model", type=str, default="runs/detect/train5/weights/best.pt", help="Path to YOLOv8 model")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--cam", type=int, default=0, help="Webcam index")
    parser.add_argument("--width", type=int, default=1280, help="Webcam width")
    parser.add_argument("--height", type=int, default=720, help="Webcam height")
    parser.add_argument("--save", action="store_true", help="Save video output")
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Load the trained model
    try:
        model = YOLO(args.model)
        print(f"Model loaded: {args.model}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Initialize webcam
    cap = cv2.VideoCapture(args.cam)
    
    # Check if webcam is opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Set webcam properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam resolution: {actual_width}x{actual_height}")
    
    # Initialize video writer if saving is enabled
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_file = f"detection_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        out = cv2.VideoWriter(output_file, fourcc, 20.0, (actual_width, actual_height))
        print(f"Recording to {output_file}")
    
    # Initialize variables for FPS calculation
    fps_array = deque(maxlen=30)  # Store last 30 FPS values for smoothing
    prev_time = time.time()
    
    # Class colors (generate once for consistent colors)
    np.random.seed(42)  # For reproducible colors
    class_colors = {}
    
    print("Press 'q' to quit, 's' to screenshot")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
            
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        fps_array.append(fps)
        avg_fps = sum(fps_array) / len(fps_array)
        prev_time = current_time
        
        # Run YOLOv8 inference on the frame
        results = model(frame, conf=args.conf)
        
        # Process each detected object
        detected_frame = frame.copy()
        
        if len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                # Get box coordinates, confidence, and class
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = results[0].names[cls_id]
                
                # Assign consistent color for this class
                if cls_id not in class_colors:
                    class_colors[cls_id] = (
                        np.random.randint(0, 255),
                        np.random.randint(0, 255),
                        np.random.randint(0, 255)
                    )
                color = class_colors[cls_id]
                
                # Draw bounding box
                cv2.rectangle(detected_frame, (x1, y1), (x2, y2), color, 2)
                
                # Create label with class name and confidence
                label = f"{cls_name}: {conf:.2f}"
                
                # Calculate label size and position
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                
                # Draw label background
                cv2.rectangle(
                    detected_frame,
                    (x1, y1 - label_height - 5),
                    (x1 + label_width, y1),
                    color,
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    detected_frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )
        
        # Add FPS and detection info to the frame
        cv2.putText(
            detected_frame,
            f"FPS: {avg_fps:.1f}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        cv2.putText(
            detected_frame,
            f"Conf threshold: {args.conf}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
        
        # Display the frame with detections
        cv2.imshow("YOLOv8 Detection", detected_frame)
        
        # Save frame if recording
        if args.save:
            out.write(detected_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('s'):  # Screenshot
            screenshot_file = f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(screenshot_file, detected_frame)
            print(f"Screenshot saved as {screenshot_file}")
        elif key == ord('+') or key == ord('='):  # Increase confidence threshold
            args.conf = min(args.conf + 0.05, 1.0)
            print(f"Confidence threshold: {args.conf:.2f}")
        elif key == ord('-'):  # Decrease confidence threshold
            args.conf = max(args.conf - 0.05, 0.0)
            print(f"Confidence threshold: {args.conf:.2f}")
    
    # Release resources
    cap.release()
    if args.save:
        out.release()
    cv2.destroyAllWindows()
    print("Detection stopped")

if __name__ == "__main__":
    main()