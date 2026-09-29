"""
Configuration and UI Styling Module for Vision-Based Object Detection & Tracking Using YOLO.
Contains theme CSS styles, model mappings, and default parameters.
"""

# App Version & Title Configuration
APP_TITLE = "Vision-Based Object Detection & Tracking Using YOLO"
APP_SUBTITLE = "Real-Time Multi-Object Detection, Continuous Tracking & Performance Analytics"
APP_ICON = "👁️"

# Standard YOLO models mapping with speed/accuracy descriptions
MODEL_OPTIONS = {
    "yolov8n.pt": "YOLOv8 Nano (Fastest - Best for Laptops / CPU)",
    "yolov8s.pt": "YOLOv8 Small (Balanced - Good Speed & Accuracy)",
    "yolov8m.pt": "YOLOv8 Medium (High Accuracy - Moderate Speed)",
    "yolov8l.pt": "YOLOv8 Large (Higher Accuracy - Requires GPU)",
    "yolov8x.pt": "YOLOv8 Extra Large (Maximum Accuracy - High GPU)"
}

# Custom CSS for impressive dark glassmorphism UI
CUSTOM_CSS = """
<style>
    /* Main Background & Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at top right, #0f172a 0%, #070a13 100%);
        color: #f8fafc;
    }
    
    /* Hide top Streamlit menu bar padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }

    /* Glassmorphism Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 229, 255, 0.15);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
    }
    
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.98rem;
        margin-top: 6px;
        font-weight: 500;
    }
    
    /* Tech Pod Cards */
    .stat-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        border-color: rgba(0, 242, 254, 0.4);
        transform: translateY(-3px);
        box-shadow: 0 8px 25px -5px rgba(0, 242, 254, 0.2);
    }
    
    .stat-val {
        font-size: 1.9rem;
        font-weight: 800;
        color: #00f2fe;
        line-height: 1.2;
    }
    
    .stat-lbl {
        font-size: 0.78rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-top: 4px;
    }
    
    /* Styled Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(15, 23, 42, 0.6);
        padding: 8px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        color: #94a3b8;
        border: none !important;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%) !important;
        color: #070a13 !important;
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.3);
    }
    
    /* Sidebar Aesthetics */
    [data-testid="stSidebar"] {
        background-color: #0b1120;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Pulsing Live Pill */
    .live-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 700;
    }
    
    .live-dot {
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 10px #10b981;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
</style>
"""
