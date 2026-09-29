"""
Vision-Based Object Detection & Tracking Using YOLO
Main Streamlit Application Dashboard with Multi-Tab Modular Architecture.
Supports Browser-Side WebRTC Webcam Streaming for Streamlit Community Cloud Deployment.
"""

import os
import time
import tempfile
import cv2
import av
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

from config import (
    APP_TITLE,
    APP_SUBTITLE,
    APP_ICON,
    MODEL_OPTIONS,
    CUSTOM_CSS
)
from detector import YOLODetector
from tracker import YOLOTracker
from analytics import AnalyticsEngine
from utils import (
    get_device,
    FPSCalculator,
    draw_annotations,
    save_screenshot,
    ProcessedVideoWriter,
    get_available_models
)

# Page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS styles
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Google Public STUN Servers for WebRTC NAT Traversal on Streamlit Cloud
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)


class YOLOVideoProcessor(VideoProcessorBase):
    """
    WebRTC Video Processor class for handling browser-side webcam streams.
    Executes YOLO detection & ByteTrack tracking on incoming video frames.
    """

    def __init__(self):
        self.tracker = None
        self.conf_thresh = 0.25
        self.iou_thresh = 0.45
        self.selected_classes = None
        self.show_boxes = True
        self.show_labels = True
        self.show_conf = True
        self.show_id = True
        self.show_hud = True
        self.fps_calc = FPSCalculator()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process incoming browser video frame through YOLO + ByteTrack pipeline."""
        img = frame.to_ndarray(format="bgr24")

        if self.tracker is not None:
            fps_val = self.fps_calc.update()

            # Execute multi-object tracking
            tracked_objects = self.tracker.track(
                frame=img,
                conf_thresh=self.conf_thresh,
                iou_thresh=self.iou_thresh,
                target_classes=self.selected_classes
            )

            # Draw annotations
            annotated_img = draw_annotations(
                frame=img,
                objects=tracked_objects,
                show_boxes=self.show_boxes,
                show_labels=show_labels,
                show_conf=show_conf,
                show_id=show_id,
                show_stats_overlay=show_hud,
                fps=fps_val
            )

            return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

        return av.VideoFrame.from_ndarray(img, format="bgr24")


@st.cache_resource
def load_yolo_tracker(model_name, tracker_type):
    """Cache detector & tracker instances to prevent reloading model weights repeatedly."""
    detector = YOLODetector(model_name=model_name)
    tracker = YOLOTracker(detector=detector, tracker_type=tracker_type)
    return tracker


def main():
    # Hardware Device Probe
    device_code, device_display = get_device()

    # Initialize Session State
    if 'analytics_engine' not in st.session_state:
        st.session_state.analytics_engine = AnalyticsEngine()
    if 'processing_active' not in st.session_state:
        st.session_state.processing_active = False

    analytics_engine = st.session_state.analytics_engine

    # ---------------------------------------------------------
    # 1. HEADER BANNER
    # ---------------------------------------------------------
    st.markdown(f"""
    <div class="hero-banner">
        <div>
            <div class="hero-title">{APP_ICON} {APP_TITLE}</div>
            <div class="hero-subtitle">{APP_SUBTITLE}</div>
        </div>
        <div class="live-pill">
            <span class="live-dot"></span> System Ready ({device_display})
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # 2. SIDEBAR CONTROL PANEL
    # ---------------------------------------------------------
    st.sidebar.markdown("### 🎛️ Control Panel")
    
    input_source = st.sidebar.radio(
        "Select Input Source:",
        ["📹 Webcam (Browser)", "🎥 Upload Video", "🖼️ Upload Image"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 Model & Tracker Settings")
    
    # Model Selection
    available_model_files = get_available_models()
    selected_model_file = st.sidebar.selectbox(
        "YOLO Model Architecture:",
        available_model_files,
        format_func=lambda x: MODEL_OPTIONS.get(x, x),
        index=0
    )
    
    # Tracker Selection
    tracker_type = st.sidebar.selectbox(
        "Object Tracker Algorithm:",
        ["ByteTrack", "BoT-SORT"],
        index=0,
        help="ByteTrack handles occlusions efficiently. BoT-SORT integrates camera motion compensation."
    )
    
    # Confidence & IoU Sliders
    conf_thresh = st.sidebar.slider(
        "Confidence Threshold:",
        min_value=0.05,
        max_value=1.00,
        value=0.25,
        step=0.05
    )
    
    iou_thresh = st.sidebar.slider(
        "IoU NMS Threshold:",
        min_value=0.10,
        max_value=1.00,
        value=0.45,
        step=0.05
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎨 Visual Display Settings")
    show_boxes = st.sidebar.checkbox("Draw Bounding Boxes", value=True)
    show_labels = st.sidebar.checkbox("Show Class Labels", value=True)
    show_conf = st.sidebar.checkbox("Show Confidence %", value=True)
    show_id = st.sidebar.checkbox("Show Persistent Tracking ID", value=True)
    show_hud = st.sidebar.checkbox("Show Top-Left HUD Badge", value=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💾 Recording & Export")
    enable_recording = st.sidebar.checkbox("Record Processed Output Video", value=False)
    
    # Load YOLO Tracker Instance
    try:
        tracker = load_yolo_tracker(selected_model_file, tracker_type.lower())
    except Exception as e:
        st.error(f"❌ Error loading YOLO model: {str(e)}")
        st.stop()

    # Class Filter Selection
    all_class_names = list(tracker.detector.get_class_names().values())
    selected_classes = st.sidebar.multiselect(
        "Filter Specific Object Classes:",
        options=all_class_names,
        default=[],
        help="Leave empty to track all 80 COCO object classes."
    )

    # ---------------------------------------------------------
    # 3. TABBED DASHBOARD LAYOUT
    # ---------------------------------------------------------
    tab_studio, tab_analytics = st.tabs([
        "👁️ Live Vision Studio",
        "📊 Real-Time Analytics"
    ])

    # =========================================================
    # TAB 1: LIVE VISION STUDIO
    # =========================================================
    with tab_studio:
        # Tech Pod Cards Row
        c1, c2, c3, c4, c5 = st.columns(5)
        
        fps_card = c1.empty()
        active_card = c2.empty()
        unique_card = c3.empty()
        model_card = c4.empty()
        device_card = c5.empty()

        def render_cards(fps=0.0, active=0, unique=0):
            fps_card.markdown(f'<div class="stat-card"><div class="stat-val">{fps:.1f}</div><div class="stat-lbl">⚡ Real-Time FPS</div></div>', unsafe_allow_html=True)
            active_card.markdown(f'<div class="stat-card"><div class="stat-val">{active}</div><div class="stat-lbl">🎯 Active Objects</div></div>', unsafe_allow_html=True)
            unique_card.markdown(f'<div class="stat-card"><div class="stat-val">{unique}</div><div class="stat-lbl">🆔 Unique Tracked</div></div>', unsafe_allow_html=True)
            model_card.markdown(f'<div class="stat-card"><div class="stat-val" style="font-size:1.3rem;">{selected_model_file.split(".")[0]}</div><div class="stat-lbl">🤖 Model</div></div>', unsafe_allow_html=True)
            device_card.markdown(f'<div class="stat-card"><div class="stat-val" style="font-size:1.3rem;">{device_code.upper()}</div><div class="stat-lbl">🖥️ Hardware</div></div>', unsafe_allow_html=True)

        render_cards()
        st.markdown("<br>", unsafe_allow_html=True)

        col_video_main, col_side_info = st.columns([3, 1])

        with col_video_main:
            video_placeholder = st.empty()
            status_placeholder = st.empty()

        with col_side_info:
            st.markdown("##### 📊 Active Class Breakdown")
            breakdown_placeholder = st.empty()
            st.markdown("##### ⚡ Quick Tools")
            btn_shot = st.button("📸 Capture Frame Screenshot", use_container_width=True)
            btn_reset_stats = st.button("🔄 Reset Session Stats", use_container_width=True)

            if btn_reset_stats:
                tracker.reset()
                analytics_engine.reset()
                st.toast("Session statistics reset!")

        # --- OPTION A: UPLOAD IMAGE ---
        if input_source == "🖼️ Upload Image":
            uploaded_image = st.file_uploader(
                "Select image file (JPG, JPEG, PNG, BMP, WEBP):",
                type=["jpg", "jpeg", "png", "bmp", "webp"]
            )

            if uploaded_image is not None:
                image_bytes = np.frombuffer(uploaded_image.read(), np.uint8)
                frame = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

                if frame is not None:
                    start_t = time.time()
                    target_cls = selected_classes if len(selected_classes) > 0 else None

                    tracked_objects = tracker.track(
                        frame=frame,
                        conf_thresh=conf_thresh,
                        iou_thresh=iou_thresh,
                        target_classes=target_cls
                    )

                    elapsed = time.time() - start_t
                    fps_val = round(1.0 / elapsed, 1) if elapsed > 0 else 0.0

                    analytics_engine.log_frame(fps_val, len(tracked_objects))

                    annotated_frame = draw_annotations(
                        frame=frame,
                        objects=tracked_objects,
                        show_boxes=show_boxes,
                        show_labels=show_labels,
                        show_conf=show_conf,
                        show_id=show_id,
                        show_stats_overlay=show_hud,
                        fps=fps_val
                    )

                    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(rgb_frame, use_container_width=True)
                    render_cards(fps_val, len(tracked_objects), len(tracker.unique_track_ids))

                    stats = tracker.get_stats()
                    class_dist = stats['class_distribution']
                    if class_dist:
                        df_dist = pd.DataFrame(list(class_dist.items()), columns=["Class", "Count"])
                        breakdown_placeholder.dataframe(df_dist, hide_index=True, use_container_width=True)

                    if btn_shot:
                        shot_file = save_screenshot(annotated_frame)
                        st.success(f"Screenshot saved: `{shot_file}`")
            else:
                video_placeholder.info("👈 Please select or upload an image file above.")

        # --- OPTION B: UPLOAD VIDEO ---
        elif input_source == "🎥 Upload Video":
            uploaded_video = st.file_uploader(
                "Select video file (MP4, AVI, MOV, MKV):",
                type=["mp4", "avi", "mov", "mkv"]
            )

            c_start, c_stop = st.columns(2)
            if c_start.button("▶️ Start Video Processing", use_container_width=True):
                st.session_state.processing_active = True
            if c_stop.button("⏹️ Stop Processing", use_container_width=True):
                st.session_state.processing_active = False

            if uploaded_video is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_video.read())
                tfile.close()

                cap = cv2.VideoCapture(tfile.name)
                if not cap.isOpened():
                    st.error("Error opening uploaded video file.")
                    os.unlink(tfile.name)
                    return

                fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                v_fps = cap.get(cv2.CAP_PROP_FPS)
                v_fps = v_fps if v_fps > 0 else 25.0

                video_writer = None
                if enable_recording:
                    video_writer = ProcessedVideoWriter(fps=v_fps, frame_size=(fw, fh))
                    video_writer.initialize((fw, fh), fps=v_fps)

                fps_calc = FPSCalculator()

                if st.session_state.processing_active:
                    status_placeholder.markdown('<div class="live-pill"><span class="live-dot"></span> Processing Stream...</div>', unsafe_allow_html=True)
                    
                    while cap.isOpened() and st.session_state.processing_active:
                        ret, frame = cap.read()
                        if not ret:
                            st.info("Reached end of video stream.")
                            st.session_state.processing_active = False
                            break

                        curr_fps = fps_calc.update()
                        target_cls = selected_classes if len(selected_classes) > 0 else None

                        tracked_objects = tracker.track(
                            frame=frame,
                            conf_thresh=conf_thresh,
                            iou_thresh=iou_thresh,
                            target_classes=target_cls
                        )

                        analytics_engine.log_frame(curr_fps, len(tracked_objects))

                        annotated_frame = draw_annotations(
                            frame=frame,
                            objects=tracked_objects,
                            show_boxes=show_boxes,
                            show_labels=show_labels,
                            show_conf=show_conf,
                            show_id=show_id,
                            show_stats_overlay=show_hud,
                            fps=curr_fps
                        )

                        if video_writer:
                            video_writer.write_frame(annotated_frame)

                        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(rgb_frame, use_container_width=True)

                        render_cards(fps_calc.get_fps(), len(tracked_objects), len(tracker.unique_track_ids))

                        stats = tracker.get_stats()
                        class_dist = stats['class_distribution']
                        if class_dist:
                            df_dist = pd.DataFrame(list(class_dist.items()), columns=["Class", "Count"])
                            breakdown_placeholder.dataframe(df_dist, hide_index=True, use_container_width=True)

                    cap.release()
                    if video_writer:
                        rec_file = video_writer.release()
                        st.success(f"🎬 Recorded video exported: `{rec_file}`")
                else:
                    video_placeholder.info("Click **▶️ Start Video Processing** above to run.")
                    cap.release()

                try:
                    os.unlink(tfile.name)
                except Exception:
                    pass
            else:
                video_placeholder.info("👈 Please upload a video file to proceed.")

        # --- OPTION C: WEBCAM (BROWSER-SIDE WEBRTC FOR STREAMLIT CLOUD) ---
        elif input_source == "📹 Webcam (Browser)":
            st.info("🌐 **Browser Webcam Stream**: Click 'START' below to grant camera access and begin live YOLOv8 + ByteTrack tracking.")

            # WebRTC Streamer component
            webrtc_ctx = webrtc_streamer(
                key="yolo-webcam-cloud",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_processor_factory=YOLOVideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

            # Pass parameters to active WebRTC video processor
            if webrtc_ctx.video_processor:
                webrtc_ctx.video_processor.tracker = tracker
                webrtc_ctx.video_processor.conf_thresh = conf_thresh
                webrtc_ctx.video_processor.iou_thresh = iou_thresh
                webrtc_ctx.video_processor.selected_classes = selected_classes if len(selected_classes) > 0 else None
                webrtc_ctx.video_processor.show_boxes = show_boxes
                webrtc_ctx.video_processor.show_labels = show_labels
                webrtc_ctx.video_processor.show_conf = show_conf
                webrtc_ctx.video_processor.show_id = show_id
                webrtc_ctx.video_processor.show_hud = show_hud

            # Display real-time stats breakdown from tracker
            stats = tracker.get_stats()
            render_cards(0.0, stats['active_objects'], stats['total_unique_tracked'])
            class_dist = stats['class_distribution']
            if class_dist:
                df_dist = pd.DataFrame(list(class_dist.items()), columns=["Class", "Count"])
                breakdown_placeholder.dataframe(df_dist, hide_index=True, use_container_width=True)

    # =========================================================
    # TAB 2: REAL-TIME ANALYTICS
    # =========================================================
    with tab_analytics:
        st.markdown("### 📈 Live Telemetry & Performance Analytics")
        
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.plotly_chart(analytics_engine.create_performance_chart(), use_container_width=True)
        with col_chart2:
            st.plotly_chart(analytics_engine.create_class_distribution_chart(tracker.get_stats()['class_distribution']), use_container_width=True)

        st.markdown("#### 📜 Session Telemetry Summary")
        stats = tracker.get_stats()
        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("Active Frame Detections", stats['active_objects'])
        col_s2.metric("Total Unique Track IDs", stats['total_unique_tracked'])
        col_s3.metric("Selected Architecture", selected_model_file)


if __name__ == "__main__":
    main()
