import streamlit as st
import cv2
import numpy as np
import time
import torch
from ultralytics import YOLO
from collections import deque
from googletrans import Translator
import tempfile
import os
import threading

# Set page configuration with custom theme
st.set_page_config(
    page_title="Sign Language Recognition",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide sidebar and other UI elements
hide_menu = """
<style>
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display:none;}
.css-1rs6os {visibility: hidden;}
div.css-1r6slb0.e1tzin5v2 {
    background-color: #f0f5ff;
    padding: 3% 3% 3% 3%;
    border-radius: 10px;
}
.css-18e3th9 {
    padding-top: 1rem;
    padding-bottom: 1rem;
    padding-left: 1rem;
    padding-right: 1rem;
}
.css-1d391kg {
    padding-top: 1rem;
    padding-right: 1rem;
    padding-bottom: 1rem;
    padding-left: 1rem;
}
</style>
"""
st.markdown(hide_menu, unsafe_allow_html=True)

# Custom CSS for better styling
st.markdown("""
<style>
/* Main page styling */
.main-header {
    color: #1E3A8A;
    font-size: 3rem;
    font-weight: 800;
    text-align: center;
    margin-bottom: 1.5rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    background: linear-gradient(90deg, #3B82F6 0%, #1E40AF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    padding: 10px;
}

.section-header {
    color: #2563EB;
    font-size: 1.5rem;
    font-weight: 600;
    margin-top: 1rem;
    margin-bottom: 0.75rem;
    border-bottom: 2px solid #BFDBFE;
    padding-bottom: 0.5rem;
}

.card {
    background-color: white;
    border-radius: 8px;
    padding: 1rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin-bottom: 1rem;
}

.detection-card {
    background: linear-gradient(145deg, #EFF6FF, #DBEAFE);
    border-radius: 10px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.1);
    margin-bottom: 1.5rem;
}

.main-detection {
    font-size: 2.5rem;
    font-weight: 700;
    color: #1E40AF;
    margin-bottom: 0.5rem;
}

.confidence {
    font-size: 1.25rem;
    color: #4B5563;
}

.language-header {
    font-size: 1.25rem;
    font-weight: 600;
    color: #1E3A8A;
    background-color: #DBEAFE;
    padding: 0.5rem;
    border-radius: 5px;
    margin-bottom: 0.5rem;
}

.stats-container {
    background: rgba(239, 246, 255, 0.8);
    border-radius: 8px;
    padding: 1rem;
    font-size: 0.9rem;
    color: #4B5563;
    margin-top: 1rem;
}

.video-container {
    border-radius: 10px;
    overflow: hidden;
    border: 2px solid #BFDBFE;
    box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1);
}

/* Detection history styling */
.history-container {
    max-height: 300px;
    overflow-y: auto;
    padding: 0.5rem;
    background-color: #F9FAFB;
    border-radius: 5px;
}

.history-item {
    padding: 0.5rem;
    margin-bottom: 0.5rem;
    border-radius: 5px;
    background-color: white;
    border-left: 4px solid #3B82F6;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}
</style>
""", unsafe_allow_html=True)

# Function to translate text to Malayalam
@st.cache_resource
def get_translator():
    return Translator()

def translate_to_malayalam(text, translator):
    try:
        translation = translator.translate(text, src='en', dest='ml')
        return translation.text
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text

# Cache translations to avoid repetitive API calls
@st.cache_data
def cached_translation(text, src='en', dest='ml'):
    translator = get_translator()
    return translate_to_malayalam(text, translator)

# Load model function with caching
@st.cache_resource
def load_model(model_path, device):
    try:
        model = YOLO(model_path)
        model.to(device)
        model.fuse()  # Fuse layers for better performance
        
        # Set model parameters for better inference speed
        if hasattr(model, 'model'):
            if hasattr(model.model, 'args'):
                # Update model parameters for faster inference
                model.model.args.update({
                    'verbose': False,
                    'half': True if device == 'cuda' else False,  # Use half precision on CUDA
                })
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    # Add a stylish title
    st.markdown("<h1 class='main-header'>SIGN LANGUAGE RECOGNITION</h1>", unsafe_allow_html=True)
    
    # Fixed configuration values (previously in sidebar)
    model_path = "runs/detect/train40/weights/best.pt"
    confidence_threshold = 0.6
    iou_threshold = 0.45
    device_option = "CUDA (if available)" if torch.cuda.is_available() else "CPU"
    
    # Set to low resolution by default
    width, height = 320, 240  # Low resolution for better performance
    process_every_n_frames = 2
    
    # Get CUDA device
    device = "cpu"
    if device_option == "CUDA (if available)" and torch.cuda.is_available():
        device = "cuda"
        # Allow more CUDA memory usage when available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Create stylish layout with cards
    col1, col2 = st.columns([5, 4])
    
    with col1:
        st.markdown("<div class='section-header'>Video Feed</div>", unsafe_allow_html=True)
        # Create a stable container for video - this helps reduce blinking
        video_placeholder = st.empty()
    
    with col2:
        st.markdown("<div class='section-header'>Current Detection</div>", unsafe_allow_html=True)
        current_detection_container = st.empty()
        
        st.markdown("<div class='section-header'>Language Translation</div>", unsafe_allow_html=True)
        # Create two columns for English and Malayalam
        en_col, ml_col = st.columns(2)
        
        with en_col:
            st.markdown("<div class='language-header'>English</div>", unsafe_allow_html=True)
            english_container = st.empty()
        
        with ml_col:
            st.markdown("<div class='language-header'>Malayalam</div>", unsafe_allow_html=True)
            malayalam_container = st.empty()
        
        # Stats at the bottom
        st.markdown("<div class='section-header'>Performance Stats</div>", unsafe_allow_html=True)
        stats_container = st.empty()
    
    # Buttons in a centered row
    button_col1, button_col2, button_col3 = st.columns([1, 2, 1])
    with button_col2:
        button_cols = st.columns(2)
        with button_cols[0]:
            start_button = st.button("Start Detection", use_container_width=True)
        with button_cols[1]:
            stop_button = st.button("Stop Detection", use_container_width=True)
    
    # Initialize session state for controlling detection
    if 'running' not in st.session_state:
        st.session_state.running = False
    
    if start_button:
        st.session_state.running = True
    
    if stop_button:
        st.session_state.running = False
        st.rerun()
    
    # Load the model right away (not inside the loop)
    if st.session_state.running:
        model = load_model(model_path, device)
        
        if model is None:
            st.error("Failed to load model. Please check the model path.")
            st.session_state.running = False
            return
        
        # Create video capture outside the loop
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce frame buffer to minimum
        
        # Check if video capture is opened successfully
        if not cap.isOpened():
            st.error("Error: Could not open webcam.")
            st.session_state.running = False
            return
        
        # Initialize translator
        translator = get_translator()
        
        # For current object tracking
        current_main_detection = None
        current_main_detection_ml = None
        current_confidence = 0
        
        # List to store recent detections for display
        recent_english_detections = []
        recent_malayalam_detections = []
        max_recent_detections = 10
        
        # FPS calculation
        fps_array = deque(maxlen=30)
        prev_time = time.time()
        frame_count = 0
        detection_times = deque(maxlen=30)
        
        # Create a placeholder for the video frame at the beginning
        video_placeholder.markdown(
            '<div class="video-container">',
            unsafe_allow_html=True
        )
        frame_placeholder = video_placeholder.empty()
        
        try:
            # Main detection loop
            while st.session_state.running:
                loop_start_time = time.time()
                
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    st.error("Error: Failed to capture frame.")
                    break
                
                # Resize frame for faster processing
                frame = cv2.resize(frame, (width, height))
                
                # Calculate FPS
                current_time = time.time()
                fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
                fps_array.append(fps)
                avg_fps = sum(fps_array) / len(fps_array)
                prev_time = current_time
                
                # Only process every Nth frame for better performance
                process_this_frame = (frame_count % process_every_n_frames == 0)
                
                # Process frame for detection
                if process_this_frame:
                    start_time = time.time()
                    results = model.predict(
                        source=frame,
                        conf=confidence_threshold,
                        iou=iou_threshold,
                        device=device,
                        verbose=False
                    )
                    detection_time = time.time() - start_time
                    detection_times.append(detection_time)
                    
                    # Process results
                    if results and len(results) > 0:
                        # Get names and labels
                        classes = results[0].names
                        boxes = results[0].boxes
                        
                        # Process only if we have results
                        if len(boxes) > 0:
                            # Find the highest confidence detection
                            highest_conf = 0
                            highest_cls_name = None
                            
                            for box in boxes:
                                # Get class ID and confidence
                                cls_id = int(box.cls[0])
                                conf = float(box.conf[0])
                                
                                # Get class name
                                if cls_id in classes:
                                    cls_name = classes[cls_id]
                                    
                                    # Keep track of highest confidence detection
                                    if conf > highest_conf:
                                        highest_conf = conf
                                        highest_cls_name = cls_name
                            
                            # If we have a new highest confidence detection
                            if highest_cls_name and (highest_cls_name != current_main_detection or highest_conf > current_confidence):
                                current_main_detection = highest_cls_name
                                current_confidence = highest_conf
                                
                                # Translate to Malayalam
                                current_main_detection_ml = cached_translation(highest_cls_name)
                                
                                # Add to recent detections
                                detection_entry = f"{current_main_detection}: {current_confidence:.2f}"
                                detection_entry_ml = f"{current_main_detection_ml}: {current_confidence:.2f}"
                                
                                # Add to the beginning of the list (newest first)
                                recent_english_detections.insert(0, detection_entry)
                                recent_malayalam_detections.insert(0, detection_entry_ml)
                                
                                # Keep only the recent detections
                                if len(recent_english_detections) > max_recent_detections:
                                    recent_english_detections.pop()
                                    recent_malayalam_detections.pop()
                        
                        # Draw detection boxes
                        annotated_frame = results[0].plot()
                    else:
                        annotated_frame = frame.copy()
                else:
                    # Skip processing but still draw any previous detections
                    try:
                        annotated_frame = results[0].plot() if 'results' in locals() and results and len(results) > 0 else frame.copy()
                    except:
                        annotated_frame = frame.copy()
                
                # Always overlay current detection info on the frame to ensure consistency
                if current_main_detection:
                    # Create text overlay for detection
                    text = f"{current_main_detection}: {current_confidence:.2f}"
                    # Draw a darker background rectangle for text visibility
                    cv2.rectangle(annotated_frame, (10, height-60), (300, height-10), (0, 0, 0), -1)
                    cv2.rectangle(annotated_frame, (10, height-60), (300, height-10), (0, 120, 255), 2)
                    # Add text with detection info
                    cv2.putText(
                        annotated_frame,
                        text,
                        (20, height-30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )
                
                # Add FPS counter with stylish overlay
                avg_detection_time = sum(detection_times) / len(detection_times) if detection_times else 0
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (10, 10), (250, 40), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
                cv2.putText(
                    annotated_frame,
                    f"FPS: {avg_fps:.1f} | Det: {avg_detection_time*1000:.1f}ms",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )
                
                # Update the video frame - fix for use_column_width warning
                frame_placeholder.image(
                    annotated_frame,
                    channels="BGR",
                    use_container_width=True,  # This is now properly applied
                    clamp=True  # Prevent pixel value warnings
                )
                
                # Update current detection with a stylish card
                if current_main_detection:
                    confidence_percentage = int(current_confidence * 100)
                    confidence_color = "#10B981" if confidence_percentage > 80 else "#F59E0B" if confidence_percentage > 50 else "#EF4444"
                    
                    current_detection_container.markdown(
                        f'''
                        <div class="detection-card">
                            <div class="main-detection">{current_main_detection}</div>
                            <div class="confidence" style="color:{confidence_color};">
                                Confidence: {confidence_percentage}%
                            </div>
                        </div>
                        ''',
                        unsafe_allow_html=True
                    )
                else:
                    # Show a placeholder when no detection is available
                    current_detection_container.markdown(
                        f'''
                        <div class="detection-card">
                            <div class="main-detection">No Detection</div>
                            <div class="confidence" style="color:#6B7280;">
                                Waiting for signs...
                            </div>
                        </div>
                        ''',
                        unsafe_allow_html=True
                    )
                
                # Format detection history without HTML tags showing
                if frame_count % 3 == 0:
                    # Create detection history as simple markdown lists instead of HTML
                    if recent_english_detections:
                        english_text = "### Recent Detections\n"
                        for det in recent_english_detections:
                            english_text += f"• {det}\n"
                        english_container.markdown(english_text)
                    else:
                        english_container.markdown("### Recent Detections\n• None yet")
                    
                    if recent_malayalam_detections:
                        malayalam_text = "### മുൻപത്തെ കണ്ടെത്തലുകൾ\n"
                        for det in recent_malayalam_detections:
                            malayalam_text += f"• {det}\n"
                        malayalam_container.markdown(malayalam_text)
                    else:
                        malayalam_container.markdown("### മുൻപത്തെ കണ്ടെത്തലുകൾ\n• ഇല്ല")
                
                # Update stats display - only update periodically
                if frame_count % 10 == 0:
                    avg_detection_time = sum(detection_times) / len(detection_times) if detection_times else 0
                    stats_container.markdown(
                        f"""
                        **Performance Stats:**
                        - **FPS:** {avg_fps:.1f}
                        - **Detection time:** {avg_detection_time*1000:.1f}ms
                        - **Device:** {device}
                        - **Resolution:** {width}x{height}
                        - **Processing:** Every {process_every_n_frames} frames
                        """
                    )
                
                # Increment frame counter
                frame_count += 1
                
                # Limit update rate to reduce blinking/flickering
                # Calculate how much time to sleep to maintain a steady framerate
                elapsed = time.time() - loop_start_time
                if elapsed < 0.033:  # Target ~30 FPS (1/30 = 0.033s per frame)
                    time.sleep(0.033 - elapsed)
        
        finally:
            # Release resources
            if cap.isOpened():
                cap.release()
            # Clean up CUDA memory if used
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    else:
        # Show a static placeholder when not running
        video_placeholder.markdown(
            """
            <div style="
                display: flex;
                justify-content: center;
                align-items: center;
                height: 300px;
                background-color: #EFF6FF;
                border-radius: 10px;
                border: 2px dashed #93C5FD;
                color: #1E40AF;
                font-size: 1.5rem;
                text-align: center;
            ">
                <div>
                    <div style="font-size: 3rem; margin-bottom: 1rem;">📷</div>
                    <div>Press "Start Detection" to activate the camera</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()