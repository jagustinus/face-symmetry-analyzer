import cv2
import dlib
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import time

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Deteksi Simetri Wajah",
    layout="centered",
    initial_sidebar_state="auto",
)

# --- Suntikkan CSS untuk Kustomisasi Tampilan ---
st.markdown("""
    <style>
    /* Mengubah warna latar belakang utama */
    .stApp {
        background-color: #f0f2f6;
    }
    /* Kustomisasi judul */
    h1 {
        color: #333;
        font-family: 'Verdana', sans-serif;
        text-align: center;
    }
    /* Kustomisasi tombol */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
        padding: 10px 24px;
    }
    </style>
    """, unsafe_allow_html=True)


# --- Fungsi Caching untuk Model (agar tidak di-load ulang setiap kali) ---
@st.cache_resource
def load_models():
    """Loads face detector and landmark predictor models."""
    try:
        face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        return face_detector, landmark_predictor
    except Exception as e:
        st.error(f"Error loading models: {e}. Pastikan file model ada di direktori yang sama.")
        return None, None

face_detector, landmark_predictor = load_models()

SYMMETRY_PAIRS = [
    (0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9),
    (17, 26), (18, 25), (19, 24), (20, 23), (21, 22),
    (36, 45), (37, 44), (38, 43), (39, 42),
    (48, 54), (49, 53), (50, 52), (60, 64), (61, 63)
]

def calculate_symmetry(landmarks_points):
    """Calculates the facial asymmetry score."""
    if not landmarks_points or len(landmarks_points) != 68:
        return 0, 0
    center_x = np.mean([p.x for p in landmarks_points])
    asymmetry_scores = [
        abs((landmarks_points[li].x - center_x) + (landmarks_points[ri].x - center_x))
        for li, ri in SYMMETRY_PAIRS
    ]
    total_asymmetry_score = sum(asymmetry_scores)
    return total_asymmetry_score / len(SYMMETRY_PAIRS), center_x

    
# --- Kelas Prosesor Video ---
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.frame_skip = 2
        self.frame_count = 0
        self.asymmetry_threshold = 3.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")

        # **PERBAIKAN FLIP: Balik gambar di sini agar konsisten untuk semua frame**
        image = cv2.flip(image, 1)

        # Logika Frame Skipping
        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0:
            # Kembalikan gambar yang sudah dibalik, tapi tanpa proses deteksi
            return av.VideoFrame.from_ndarray(image, format="bgr24")

        # Logika Deteksi (hanya untuk frame yang tidak dilewati)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if face_detector is not None:
            faces = face_detector.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 192, 203), 2)
                dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
                
                if landmark_predictor is not None:
                    landmarks = landmark_predictor(gray, dlib_rect)
                    landmark_points = [landmarks.part(n) for n in range(68)]

                    for pt in landmark_points:
                        cv2.circle(image, (pt.x, pt.y), 2, (135, 206, 250), -1)

                    asymmetry_score, center_x = calculate_symmetry(landmark_points)
                    label = "Simetris" if asymmetry_score < self.asymmetry_threshold else "Kurang Simetris"
                    color = (0, 255, 0) if label == "Simetris" else (0, 0, 255)

                    cv2.putText(image, label, (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    cv2.putText(image, f"Skor Asimetri: {asymmetry_score:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.line(image, (int(center_x), y), (int(center_x), y + h), (255, 255, 0), 1)

        return av.VideoFrame.from_ndarray(image, format="bgr24")

# --- UI Utama Streamlit ---
st.title("Analisis Simetri Wajah")
st.markdown("Aplikasi ini akan menganalisis tingkat simetri wajah Anda secara *real-time*. Wajah yang lebih simetris seringkali dianggap lebih menarik secara universal.")

# Sidebar untuk kontrol
with st.sidebar:
    st.header("Pengaturan")
    st.info("Aplikasi ini membutuhkan akses ke kamera Anda.")
    threshold = st.slider("Ambang Batas Simetri", min_value=1.0, max_value=5.0, value=3.0, step=0.1)

# Konfigurasi RTC
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

webrtc_ctx = webrtc_streamer(
    key="face-analysis",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
    async_processing=True,
)

if webrtc_ctx.video_processor:
    webrtc_ctx.video_processor.asymmetry_threshold = threshold

st.sidebar.markdown("---")
st.sidebar.markdown("Dibuat dengan ❤️ menggunakan Streamlit.")
