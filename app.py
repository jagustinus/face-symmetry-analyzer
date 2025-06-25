# --- app.py ---
# import cv2
# import dlib
# import numpy as np
# import streamlit as st
# from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration
# import av
# import threading

import asyncio
import sys
import platform
import logging

logging.getLogger("aioice").setLevel(logging.WARNING)
logging.getLogger("aiortc").setLevel(logging.WARNING)

if platform.system() == "Linux":
    try:
        # Set a more compatible event loop policy for Docker
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
        # Create and set a new event loop for the main thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    except Exception as e:
        print(f"Event loop setup warning: {e}")

import cv2
import dlib
import numpy as np
import streamlit as st
from streamlit_webrtc import (
    webrtc_streamer,
    VideoTransformerBase,
    WebRtcMode,
    RTCConfiguration,
)
import av
import threading


# --- Konfigurasi halaman dan tema ---
st.set_page_config(page_title="Tes Wajah Keren Mu Menurut AI", layout="wide")

st.markdown(
    """
<style>
html, body, [class*="st-"] {
    font-family: 'Segoe UI', sans-serif;
    background-color: #0E1117;
    color: #FAFAFA;
}
.block-container {
    padding: 3rem 3rem 1rem 3rem;
}
.title {
    text-align: center;
    font-size: 2.5rem;
    margin-bottom: 0.3rem;
    margin-top: 2rem;
}
.subtitle {
    text-align: center;
    color: #8A919E;
    margin-bottom: 2.5rem;
}
.stTextInput>div>input {
    border-radius: 8px;
    padding: 10px;
}
.stButton>button {
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 600;
}
.leaderboard-entry {
    padding: 8px 0;
    border-bottom: 1px solid #31333F;
    display: flex;
    justify-content: space-between;
}
.leaderboard-entry span.rank {
    font-weight: 700;
    color: #00C9A7;
}
/* Tabs UI custom */
.css-1hynsf2 .stTabs [data-baseweb="tab"] {
    background-color: #262730;
    border-radius: 10px 10px 0 0;
    margin-right: 2px;
    padding: 10px;
    font-weight: 600;
    color: #ccc;
    border-bottom: 3px solid transparent;
    transition: all 0.3s ease;
}
.css-1hynsf2 .stTabs [aria-selected="true"] {
    color: white;
    border-bottom: 3px solid #00C9A7;
    background-color: #333;
}
.css-1hynsf2 .stTabs [data-baseweb="tab"]:hover {
    background-color: #3a3a3a;
}
</style>
""",
    unsafe_allow_html=True,
)

# --- Inisialisasi state ---
if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = []
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None


# --- Load model ---
@st.cache_resource
def load_models():
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    return detector, predictor


face_detector, landmark_predictor = load_models()

# --- Simetri calculation ---
SYMMETRY_PAIRS = [
    (0, 16),
    (1, 15),
    (2, 14),
    (3, 13),
    (4, 12),
    (5, 11),
    (6, 10),
    (7, 9),
    (17, 26),
    (18, 25),
    (19, 24),
    (20, 23),
    (21, 22),
    (36, 45),
    (37, 44),
    (38, 43),
    (39, 42),
    (48, 54),
    (49, 53),
    (50, 52),
    (60, 64),
    (61, 63),
]


def calculate_symmetry(landmarks):
    center_x = np.mean([p.x for p in landmarks])
    score = np.mean(
        [
            abs((landmarks[l].x - center_x) + (landmarks[r].x - center_x))
            for l, r in SYMMETRY_PAIRS
        ]
    )
    return score, center_x


# --- Analisis gambar ---
def analyze_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
    if len(faces) == 0:
        return {"face_found": False}

    (x, y, w, h) = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
    rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
    landmarks = landmark_predictor(gray, rect)
    points = [landmarks.part(i) for i in range(68)]

    score, cx = calculate_symmetry(points)

    output = image.copy()
    for pt in points:
        cv2.circle(output, (pt.x, pt.y), 2, (0, 255, 255), -1)
    cv2.line(output, (int(cx), y), (int(cx), y + h), (0, 255, 0), 1)
    cv2.rectangle(output, (x, y), (x + w, y + h), (255, 192, 203), 2)

    return {"face_found": True, "score": score, "image": output}


# --- Kelas VideoProcessor dengan live detection ---
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.latest_frame = None
        self.lock = threading.Lock()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 192, 203), 2)
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            landmarks = landmark_predictor(gray, rect)
            points = [landmarks.part(i) for i in range(68)]

            for pt in points:
                cv2.circle(img, (pt.x, pt.y), 2, (0, 255, 255), -1)

            score, cx = calculate_symmetry(points)
            cv2.putText(
                img,
                f"Skor: {score:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.line(img, (int(cx), y), (int(cx), y + h), (0, 255, 0), 1)

        with self.lock:
            self.latest_frame = img.copy()

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# --- Judul dan deskripsi ---
st.markdown("<div class='title'>Test Keren Mu menurut AI</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Analisis Wajah Keren berdasarkan simetri proporsional</div>",
    unsafe_allow_html=True,
)

# --- Layout utama ---
col1, col2 = st.columns([1.05, 0.95])

# --- Kolom kiri: kamera dan input ---
with col1:
    st.subheader("📸 Live Detection")

    rtc_config = RTCConfiguration(
        {
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
                {"urls": ["stun:stun3.l.google.com:19302"]},
                {"urls": ["stun:stun4.l.google.com:19302"]},
                {
                    "urls": "turn:openrelay.metered.ca:80",
                    "username": "openrelayproject",
                    "credential": "openrelayproject",
                },
                {
                    "urls": "turn:openrelay.metered.ca:443",
                    "username": "openrelayproject",
                    "credential": "openrelayproject",
                },
            ]
        }
    )

    try:
        ctx = webrtc_streamer(
            key="live",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    except Exception as e:
        st.error(
            "WebRTC tidak dapat dimulai. Silakan refresh halaman atau gunakan browser yang mendukung WebRTC."
        )
        st.info(
            "Untuk sementara, Anda dapat menggunakan upload foto sebagai alternatif."
        )
        ctx = None

    col_input, col_btn = st.columns([3, 1])
    with col_input:
        username = st.text_input(
            "Masukkan Username",
            label_visibility="collapsed",
            placeholder="cth: Maglon Martino / Informatika",
        )
    with col_btn:
        analyze = st.button("Analisis")

    if analyze and ctx.video_processor and username:
        with ctx.video_processor.lock:
            frame = ctx.video_processor.latest_frame

        if frame is not None:
            result = analyze_image(frame)
            st.session_state.analysis_result = result
            if result["face_found"]:
                st.session_state.leaderboard.append(
                    {"username": username, "score": result["score"]}
                )
                st.session_state.leaderboard.sort(key=lambda x: x["score"])

# --- Kolom kanan: tab Peringkat & Analisis ---
with col2:
    tab1, tab2 = st.tabs(["🏆 Peringkat", "📊 Analisis"])

    with tab1:
        st.subheader("Ranking Paling Keren Saat Ini")
        if not st.session_state.leaderboard:
            st.info("Belum ada data, ayo mulai analisis!")
        else:
            for i, d in enumerate(st.session_state.leaderboard):
                st.markdown(
                    f"<div class='leaderboard-entry'>"
                    f"<span class='rank'>#{i+1}</span> <span>{d['username']}</span> <span>{d['score']:.2f}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    with tab2:
        st.subheader("Hasil Analisis Wajah")
        result = st.session_state.analysis_result
        if result and result["face_found"]:
            st.image(
                result["image"],
                channels="BGR",
                caption="Hasil deteksi wajah & landmark",
            )
            st.metric("Skor Simetri", f"{result['score']:.2f}")
            score = result["score"]
            if score < 2.0:
                st.success(
                    "✨ Sangat KERENN!!! Sangat simetris! Wajahmu luar biasa proporsional ."
                )
            elif score < 4.0:
                st.info(
                    "😊 KERENN, Wajah Simetris dan menarik! Ini nilai wajah yang bagus."
                )
            elif score < 6.5:
                st.warning("😉 Unik dan punya ciri khas. Simetri tetap baik.")
            else:
                st.error("👍 Simetri Kamu Cukup rendah, Tapi pesona tidak tergantikan!")
        else:
            st.info("Klik tombol analisis setelah wajah terdeteksi.")
