# Sign Language Recognition and Translation App

This project is a real-time Sign Language Detection and Translation web application built using **YOLOv8** for hand gesture recognition, **Streamlit** for the interface, and **Google Translate API** for translating detected signs into Malayalam.

## 🔍 Features

- Real-time webcam input to detect hand gestures representing sign language.
- Integrated YOLOv8 object detection model for fast and accurate gesture recognition.
- Translates detected English class labels into Malayalam using Google Translate.
- Displays recent detected words and translations.
- Shows performance stats like FPS, average detection time, and confidence levels.
- User-friendly interface with start/stop functionality.

## 🧰 Technologies Used

- [YOLOv8](https://github.com/ultralytics/ultralytics) (Ultralytics)
- [Streamlit](https://streamlit.io/)
- [OpenCV](https://opencv.org/)
- [Google Translate API (translatepy)](https://pypi.org/project/translatepy/)
- Python Libraries: `cv2`, `ultralytics`, `streamlit`, `translatepy`, `numpy`, etc.

## 🚀 Getting Started

### 1. Clone the Repository

git clone https://github.com/yourusername/sign-language-translator.git
cd sign-language-translator
2. Install Requirements
Make sure you have Python 3.8+ installed.

bash
Copy
Edit
pip install -r requirements.txt
requirements.txt should include:
nginx
Copy
Edit
ultralytics
opencv-python
streamlit
translatepy
numpy
3. Run the Application
bash
Copy
Edit
streamlit run app.py
📂 Project Structure
bash
Copy
Edit
├── app.py                  # Main Streamlit application
├── README.md
├── requirements.txt
└── yolov8_model.pt         # Your trained YOLOv8 model for sign detection
📸 Demo

Replace with your demo image or GIF showing real-time detection and translation.

✍️ Customization
Model Training: Use Ultralytics YOLOv8 to train your own dataset if needed.

Translation: You can change the target language in translatepy.Translator() if you want support for other languages.

🧠 Future Improvements
Add support for sentence construction from continuous signs.

Support multiple languages and dialects.

Speech output for detected signs (Text-to-Speech integration).

Mobile deployment using Streamlit Sharing or HuggingFace Spaces.

🙏 Acknowledgments
Ultralytics for YOLOv8.

translatepy for translation support.

Streamlit for building the UI effortlessly
