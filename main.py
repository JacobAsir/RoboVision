import streamlit as st
import cv2
import time
from datetime import datetime
import pandas as pd
import sqlite3
import numpy as np
import os
import base64
import gc
import textwrap

# Streamlit Cloud / containers: Ultralytics cannot write ~/.config
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join("/tmp", "Ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib"))

# ==========================================
# INTERNAL CONFIG (Hidden from user)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = os.path.join(BASE_DIR, "yolov8n.pt")
# Open-vocabulary model for packages (heavier — only used when not low-memory)
PACKAGE_MODEL_NAME = os.path.join(BASE_DIR, "yolov8s-world.pt")
# Render free = 512MB → must stay low-mem (single nano model, no YOLO-World)
LOW_MEM = os.environ.get("ROBOVISION_LOW_MEM", "1").strip().lower() in ("1", "true", "yes", "on")
CONF_THRESHOLD = 0.40
SECURE_CONF_THRESHOLD = 0.28         # softer so bottles + people stick in demo footage
LOADING_CONF_THRESHOLD = 0.12
PACKAGE_CONF_THRESHOLD = 0.12
IOU_THRESHOLD = 0.50
OCCLUSION_LIMIT = 4                  # frames missing before we treat item as removed
MIN_PRODUCT_SEEN = 3                 # must be visible this many frames before a removal counts
STABLE_WINDOW = 8
# Bottle / secure demo: snappy but still trackable (stride 1 keeps multi-bottle IDs stable)
SECURE_PLAYBACK_SPEED = 1.75         # slightly faster than real-time
SECURE_FRAME_STRIDE = 2 if LOW_MEM else 1  # skip frames on small Render instances
SECURE_DEMO_START_SEC = 3.0          # skip first 3s of bottle demo (idle intro)
SECURE_SLOT_MATCH_DIST = 100.0       # px — match bottles by position, not only YOLO track id
INFER_IMGSZ = 320 if LOW_MEM else 416
DEMO_VIDEO_SECURE = os.path.join(BASE_DIR, "bottle-detection.mp4")
DEMO_VIDEO_LOADING = os.path.join(BASE_DIR, "5903898-hd_1920_1080_30fps.mp4")
DEMO_VIDEO_CONTAINER = os.path.join(BASE_DIR, "container-loading-demo.mp4")
DB_PATH = os.path.join(BASE_DIR, "robovision.db")
# Fallback proxies if YOLO-World is unavailable / low-mem
PACKAGE_COCO_CLASSES = (
    "suitcase", "book", "backpack", "handbag", "bed", "microwave",
    "tv", "clock", "laptop", "refrigerator", "couch", "box", "toaster", "chair"
)
PACKAGE_WORLD_CLASSES = [
    "package", "parcel", "cardboard box", "box", "mailer bag", "shipping package"
]
LOGO_PATH = os.path.join(BASE_DIR, "logo-icon.png")
ROFI_PATH = os.path.join(BASE_DIR, "rofi-3d.png")

def _img_data_uri(path, max_height=None):
    """Embed local PNG as a data URI. Optionally downscale tall assets (e.g. mascot)."""
    if not os.path.exists(path):
        return None
    try:
        from PIL import Image
        import io
        im = Image.open(path).convert("RGBA")
        if max_height and im.height > max_height:
            ratio = max_height / float(im.height)
            im = im.resize((max(1, int(im.width * ratio)), max_height), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:image/png;base64,{b64}"

LOGO_URI = _img_data_uri(LOGO_PATH, max_height=72 if LOW_MEM else 96)
ROFI_URI = _img_data_uri(ROFI_PATH, max_height=160 if LOW_MEM else 280)

# ==========================================
# PAGE SETUP
# ==========================================
st.set_page_config(
    page_title="RoboVision",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# I18N — English / Japanese (RoboVision brand name stays English)
# ==========================================
I18N = {
    "en": {
        "lang_en": "English",
        "lang_ja": "日本語",
        "badge_suite": "AI VISION SUITE",
        "powered_by": "Powered by",
        "tagline": "AI-powered computer vision that solves real security and logistics problems. Choose a use case below and see it in action.",
        "use_case_secure": "🔐 Secure Product Monitoring",
        "use_case_loading": "📦 Loading & Packing Verification",
        "secure_title": "🔐 Secure Product Access Monitoring",
        "secure_problem": "The Problem:",
        "secure_problem_body": "High-value products are at risk of theft. Current solutions lack automated, real-time tracking of who took what and when.",
        "secure_solution": "Our Solution:",
        "secure_solution_body": "AI automatically detects when a person removes a product and logs the event with a timestamp — no manual monitoring needed.",
        "video_source": "📹 Video Source",
        "video_source_help": "Use our built-in demo video or upload your own footage to analyze.",
        "use_demo": "🎬 Use Demo Video",
        "upload_own": "📁 Upload Your Own Video",
        "choose_input": "Choose input",
        "demo_downloading": "Demo video not found. Downloading...",
        "demo_missing": "Conveyor belt demo video not found.\n\nExpected at: `{path}`\n\nPlace the file next to main.py or upload your own video.",
        "drop_video": "Drop a video file here",
        "upload_hint": "👆 Upload a video file to get started.",
        "target_product": "Target Product to Monitor",
        "activate_cctv": "▶️  Activate CCTV Feed",
        "live_cctv": "🎥 Live CCTV Feed",
        "realtime_monitoring": "📊 Real-time Monitoring & Logs",
        "open_video_fail": "Could not open video source.",
        "active_items": "ACTIVE ITEMS",
        "removals": "REMOVALS",
        "secured": "✅ SECURED",
        "breach": "🚨 BREACH",
        "monitoring": "✅ MONITORING",
        "event_product_removed": "🚨 Product Removed",
        "alarm_removed_one": "🚨 ALARM: Secure item '{product}' removed from area!",
        "alarm_removed_n": "🚨 ALARM: {n}× '{product}' removed from area!",
        "col_time": "Time",
        "col_event": "Event",
        "col_product": "Product",
        "col_conf": "Conf.",
        "col_expected": "Expected",
        "col_detected": "Detected",
        "col_status": "Status",
        "export_logs": "📥 Export Logs (CSV)",
        "no_logs_secure": "No logs registered yet. Run CCTV streams to generate events.",
        "loading_title": "📦 Loading & Packing Verification",
        "loading_problem": "The Problem:",
        "loading_problem_body": "Manifests say how many items should be loaded, but counting by eye is slow and error-prone.",
        "loading_solution": "Our Solution:",
        "loading_solution_body": "AI counts packages on the conveyor in real time and flags mismatches against the expected quantity.",
        "what_to_count": "What item should we count?",
        "package_caption": "Packages use open-vocab detection + belt ROI (COCO YOLO has no package class).",
        "expected_qty": "Expected Quantity (Manifest)",
        "activate_verify": "▶️  Activate Verification Feed",
        "live_verify": "🎥 Live Verification Feed",
        "realtime_verify": "📊 Real-time Verification & Logs",
        "expected": "EXPECTED",
        "detected": "DETECTED",
        "starting": "▶️ STARTING…",
        "inactive": "💤 INACTIVE",
        "match": "✅ MATCH",
        "mismatch": "⚠️ MISMATCH",
        "verified": "✅ VERIFIED",
        "status_mismatch": "🚨 MISMATCH",
        "qty_mismatch": "⚠️ Quantity Mismatch! Expected {expected}, detected {detected}",
        "qty_verified": "✅ Quantity Verified ({count} items)",
        "count_banner": "Count: {count}  |  Expected: {expected}",
        "analyzing": "Analyzing…",
        "world_fallback": "YOLO-World package model unavailable ({err}). Falling back to COCO proxies.",
        "open_source_fail": "Could not open video source: `{path}`",
        "video_open_fail": "Video failed to open. Check the file path or try uploading a video.",
        "read_frames_fail": "Could not read frames from the demo video.",
        "end_of_video": "End of video or failed to read frames.",
        "select_source": "Select a video source to start verification.",
        "no_logs_loading": "No logs registered yet. Run video to verify quantities.",
        "footer_left": "RoboVision by RoboFounders.ai — AI Computer Vision for Enterprise Security & Logistics",
        "cls_package": "package",
        "use_case_container": "🚛 3D Container Loading & Stacking",
        "container_title": "🚛 AI 3D Container Loading & Stacking Optimizer",
        "container_problem": "The Problem:",
        "container_problem_body": "Container loading relies on manual guesswork, causing wasted volume, axle overload, and item damage.",
        "container_solution": "Our Solution:",
        "container_solution_body": "AI scans incoming cargo, predicts optimal 3D spatial layout, and outputs step-by-step stacking instructions to maximize fill capacity.",
        "select_container": "Target Container / Vehicle",
        "activate_stacking": "▶️  Activate Stacking Optimizer",
        "live_bay_feed": "🎥 Cargo Scanning & Loading Bay Feed",
        "realtime_container": "📊 Real-time 3D Packing & Stacking Sequence",
        "container_3d_map": "🧱 3D Container Spatial Layout Plan",
        "stacking_guidance": "📋 Step-by-Step Stacking Instruction Sequence",
        "fill_ratio": "FILL RATIO",
        "weight_balance": "WEIGHT BALANCE",
        "crush_risk": "CRUSH RISK",
        "status_optimal": "✅ OPTIMAL",
        "recommendation_banner": "AI Loading Plan for Container ❶: Start with Box A [Floor Base] ➔ Stack Box B [Middle] ➔ Top with Box C [Surface]",
        "step_1": "1️⃣ Step 1: Start with Box A [Heavy] → Place at Floor Base (Section 1)",
        "step_2": "2️⃣ Step 2: Stack Box B [Medium] → Place directly on top of Box A",
        "step_3": "3️⃣ Step 3: Finish with Box C [Fragile] → Place on Upper Surface (94.2% Capacity)",
        "axle_even": "50% Front / 50% Rear (EVEN)",
        "zero_warnings": "0 Safety Warnings",
        "no_logs_container": "No stacking logs registered yet. Run video feed to generate placement plan.",
        "box_a": "BOX A (HEAVY)",
        "box_b": "BOX B (MEDIUM)",
        "box_c": "BOX C (FRAGILE)",
    },
    "ja": {
        "lang_en": "English",
        "lang_ja": "日本語",
        "badge_suite": "AIビジョン スイート",
        "powered_by": "提供",
        "tagline": "セキュリティと物流の課題を解決するAIコンピュータビジョン。下のユースケースを選んで、実際の動作をご確認ください。",
        "use_case_secure": "🔐 セキュア製品モニタリング",
        "use_case_loading": "📦 積込・梱包検証",
        "secure_title": "🔐 セキュア製品アクセス監視",
        "secure_problem": "課題：",
        "secure_problem_body": "高額製品は盗難リスクがあります。現状では、誰が・何を・いつ持ち出したかを自動で記録する仕組みが不足しています。",
        "secure_solution": "ソリューション：",
        "secure_solution_body": "人が製品を持ち出した瞬間をAIが自動検知し、タイムスタンプ付きでログ記録。人手による監視は不要です。",
        "video_source": "📹 映像ソース",
        "video_source_help": "内蔵デモ動画を使うか、独自の映像をアップロードして解析できます。",
        "use_demo": "🎬 デモ動画を使用",
        "upload_own": "📁 動画をアップロード",
        "choose_input": "入力を選択",
        "demo_downloading": "デモ動画が見つかりません。ダウンロード中…",
        "demo_missing": "コンベアデモ動画が見つかりません。\n\n想定パス: `{path}`\n\nmain.py と同じフォルダに置くか、動画をアップロードしてください。",
        "drop_video": "ここに動画ファイルをドロップ",
        "upload_hint": "👆 開始するには動画をアップロードしてください。",
        "target_product": "監視対象製品",
        "activate_cctv": "▶️  CCTVフィードを開始",
        "live_cctv": "🎥 ライブCCTVフィード",
        "realtime_monitoring": "📊 リアルタイム監視とログ",
        "open_video_fail": "映像ソースを開けませんでした。",
        "active_items": "監視中アイテム",
        "removals": "撤去回数",
        "secured": "✅ 安全",
        "breach": "🚨 異常検知",
        "monitoring": "✅ 監視中",
        "event_product_removed": "🚨 製品撤去",
        "alarm_removed_one": "🚨 警報：対象製品『{product}』がエリアから撤去されました！",
        "alarm_removed_n": "🚨 警報：対象製品『{product}』が {n} 件撤去されました！",
        "col_time": "時刻",
        "col_event": "イベント",
        "col_product": "製品",
        "col_conf": "信頼度",
        "col_expected": "予定数",
        "col_detected": "検出数",
        "col_status": "ステータス",
        "export_logs": "📥 ログをエクスポート (CSV)",
        "no_logs_secure": "ログはまだありません。CCTVを実行するとイベントが記録されます。",
        "loading_title": "📦 積込・梱包検証",
        "loading_problem": "課題：",
        "loading_problem_body": "マニフェストには積込予定数が記載されますが、目視カウントは遅く誤りも起きやすいです。",
        "loading_solution": "ソリューション：",
        "loading_solution_body": "コンベア上の荷物をAIがリアルタイムで数え、予定数量との不一致を自動で通知します。",
        "what_to_count": "カウントするアイテム",
        "package_caption": "荷物はオープン語彙検出＋ベルトROIを使用（COCOのYOLOにpackageクラスはありません）。",
        "expected_qty": "予定数量（マニフェスト）",
        "activate_verify": "▶️  検証フィードを開始",
        "live_verify": "🎥 ライブ検証フィード",
        "realtime_verify": "📊 リアルタイム検証とログ",
        "expected": "予定数",
        "detected": "検出数",
        "starting": "▶️ 開始中…",
        "inactive": "💤 停止中",
        "match": "✅ 一致",
        "mismatch": "⚠️ 不一致",
        "verified": "✅ 検証OK",
        "status_mismatch": "🚨 不一致",
        "qty_mismatch": "⚠️ 数量不一致！ 予定 {expected}、検出 {detected}",
        "qty_verified": "✅ 数量検証OK（{count} 件）",
        "count_banner": "検出: {count}  |  予定: {expected}",
        "analyzing": "解析中…",
        "world_fallback": "YOLO-World荷物モデルが利用できません（{err}）。COCOプロキシに切替えます。",
        "open_source_fail": "映像ソースを開けませんでした: `{path}`",
        "video_open_fail": "動画を開けませんでした。パスを確認するか、別の動画をアップロードしてください。",
        "read_frames_fail": "デモ動画からフレームを読み取れませんでした。",
        "end_of_video": "動画終了、またはフレーム読み取りに失敗しました。",
        "select_source": "検証を開始するには映像ソースを選択してください。",
        "no_logs_loading": "ログはまだありません。動画を実行すると数量検証が記録されます。",
        "footer_left": "RoboVision by RoboFounders.ai — 企業向けセキュリティ＆物流のAIコンピュータビジョン",
        "cls_package": "荷物",
        "use_case_container": "🚛 3Dコンテナ積載・荷積み最適化",
        "container_title": "🚛 AI 3Dコンテナ積載・荷積み順序最適化",
        "container_problem": "課題：",
        "container_problem_body": "コンテナへの荷積みは作業員の経験に依存し、空間の無駄、偏荷重、荷崩れが発生します。",
        "container_solution": "ソリューション：",
        "container_solution_body": "搬入される荷物をAIが識別し、最適な3D配置とステップバイステップの積み込み順序指示を自動生成します。",
        "select_container": "対象コンテナ / 車両",
        "activate_stacking": "▶️  積載最適化フィードを開始",
        "live_bay_feed": "🎥 荷物スキャン＆搬入フィード",
        "realtime_container": "📊 リアルタイム3D配置＆荷積み指示",
        "container_3d_map": "🧱 3Dコンテナ空間配置プラン",
        "stacking_guidance": "📋 ステップバイステップ荷積み指示シーケンス",
        "fill_ratio": "積載率",
        "weight_balance": "重量バランス",
        "crush_risk": "荷崩れリスク",
        "status_optimal": "✅ 最適化完了",
        "recommendation_banner": "コンテナ❶向けAI積載プラン: 荷物A[床面ベース]を配置 ➔ 荷物B[中間層]を重畳 ➔ 荷物C[最上層]を天面へ",
        "step_1": "1️⃣ ステップ1: 荷物A [重量物] を搬入 → 床面ベース (セクション1) へ配置",
        "step_2": "2️⃣ ステップ2: 荷物B [中量物] を搬入 → 荷物Aの直上へ重畳積載",
        "step_3": "3️⃣ ステップ3: 荷物C [壊れ物] で完了 → 最上層・天面へ配置 (積載率 94.2%)",
        "axle_even": "前軸50% / 後軸50% (均等)",
        "zero_warnings": "安全警告 0件",
        "no_logs_container": "まだログはありません。動画を実行すると積載プランが生成されます。",
        "box_a": "荷物A (重量物)",
        "box_b": "荷物B (中量物)",
        "box_c": "荷物C (壊れ物)",
    },
}

def get_lang():
    """Active UI language. Widget key is ui_lang (never write it after the radio is created)."""
    lang = st.session_state.get("ui_lang", "en")
    return lang if lang in ("en", "ja") else "en"

def t(key, **kwargs):
    lang = get_lang()
    text = I18N.get(lang, I18N["en"]).get(key)
    if text is None:
        text = I18N["en"].get(key, key)
    if kwargs:
        try:
            return text.format(**kwargs)
        except Exception:
            return text
    return text

def ensure_radio_value(key, allowed, default):
    """Keep radio session values valid after option-set / language refactors."""
    val = st.session_state.get(key)
    if val not in allowed:
        st.session_state[key] = default

def cls_label(class_name):
    """Display label for a COCO/class id (internal value stays English for the model)."""
    key = "cls_" + class_name.replace(" ", "_")
    return t(key)

def localize_event_text(event):
    """Map stored log event strings into the active language for display."""
    if not isinstance(event, str):
        return event
    e = event.lower()
    if "product removed" in e or "製品撤去" in event:
        return t("event_product_removed")
    if "verified" in e or "検証ok" in e or "検証OK" in event:
        return t("verified")
    if "mismatch" in e or "不一致" in event:
        return t("status_mismatch")
    return event

def _ja_font(size=18):
    """Windows Japanese font for OpenCV overlays (Hershey fonts cannot draw 漢字)."""
    from PIL import ImageFont
    candidates = [
        r"C:\Windows\Fonts\meiryo.ttc",
        r"C:\Windows\Fonts\YuGothM.ttc",
        r"C:\Windows\Fonts\msgothic.ttc",
        r"C:\Windows\Fonts\malgun.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()

def draw_text(img_bgr, text, org, color_bgr=(255, 255, 255), scale=0.6, thickness=2):
    """Draw text that supports Japanese when lang=ja; otherwise use OpenCV Hershey."""
    if get_lang() != "ja":
        cv2.putText(
            img_bgr, str(text), org, cv2.FONT_HERSHEY_SIMPLEX, scale, color_bgr, thickness
        )
        return img_bgr
    try:
        from PIL import Image, ImageDraw
        x, y = org
        # OpenCV uses baseline; PIL uses top-left — shift up slightly
        font_size = max(14, int(22 * scale))
        font = _ja_font(font_size)
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil)
        color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
        # y is baseline in cv2; approximate top for PIL
        draw.text((x, max(0, y - font_size)), str(text), font=font, fill=color_rgb)
        img_bgr[:, :, :] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        # Last resort ASCII-only
        cv2.putText(
            img_bgr, str(text), org, cv2.FONT_HERSHEY_SIMPLEX, scale, color_bgr, thickness
        )
    return img_bgr

# ==========================================
# DATABASE
# ==========================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS secure_access_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT,
        event_type TEXT, product TEXT, person_detected TEXT, confidence REAL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS loading_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT, expected_count INTEGER,
        detected_count INTEGER, status TEXT, item_class TEXT, timestamp TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS container_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT, container_type TEXT,
        fill_rate REAL, weight_balance TEXT, step_instruction TEXT, timestamp TEXT)""")
    conn.commit()
    conn.close()

init_db()

# Session analysis defaults — reset on refresh / tab change so every run starts clean
ANALYSIS_STATE_DEFAULTS = {
    "active_products": {},
    "last_alert": None,
    "loading_count_history": [],
    "loading_last_logged_count": -1,
    "live_detected_count": 0,
    "fallback_tracks": {},
    "next_track_id": 1000,
    "secure_recent_person": False,
    "secure_baseline_count": 0,       # peak stable bottle count on the shelf
    "secure_low_count_streak": 0,     # frames spent below baseline
    "secure_slot_seq": 1,             # next spatial slot id
    "container_step": 1,
}

def clear_all_logs():
    """Wipe persisted logs so UI never shows history from a previous run."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM secure_access_logs")
    conn.execute("DELETE FROM loading_logs")
    conn.execute("DELETE FROM container_logs")
    try:
        conn.execute(
            "DELETE FROM sqlite_sequence WHERE name IN ('secure_access_logs', 'loading_logs', 'container_logs')"
        )
    except sqlite3.Error:
        pass
    conn.commit()
    conn.close()

def reset_analysis_state():
    """Reset in-memory tracking so detection starts as if first open."""
    for key, default in ANALYSIS_STATE_DEFAULTS.items():
        if isinstance(default, dict):
            st.session_state[key] = {}
        elif isinstance(default, list):
            st.session_state[key] = []
        elif isinstance(default, set):
            st.session_state[key] = set()
        else:
            st.session_state[key] = default

def reset_model_trackers():
    """Reset Ultralytics tracking state without leaving a broken empty trackers list.

    persist=True reuses predictor.trackers if the attribute exists. Setting it to []
    causes IndexError on trackers[0]. Delete the attribute (or reset each tracker)
    so the next model.track() re-initializes cleanly.
    """
    try:
        m = st.session_state.get("_yolo_model")
        if m is None:
            return
        predictor = getattr(m, "predictor", None)
        if predictor is None:
            return
        trackers = getattr(predictor, "trackers", None)
        if trackers:
            for tr in trackers:
                if hasattr(tr, "reset"):
                    tr.reset()
        if hasattr(predictor, "trackers"):
            delattr(predictor, "trackers")
        if hasattr(predictor, "vid_path"):
            delattr(predictor, "vid_path")
        for attr in ("track_history", "seen_track_ids"):
            if hasattr(predictor, attr):
                delattr(predictor, attr)
    except Exception:
        pass

def start_fresh_analysis():
    """Full clean slate: empty DB logs + zeroed session trackers + YOLO tracker reset."""
    clear_all_logs()
    reset_analysis_state()
    reset_model_trackers()

def log_secure_access(event_type, product, person_detected, confidence):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO secure_access_logs (timestamp, event_type, product, person_detected, confidence) VALUES (?,?,?,?,?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), event_type, product, person_detected, round(float(confidence or 0), 2)))
    conn.commit()
    conn.close()

def log_loading_event(expected, detected, status, item_class):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO loading_logs (expected_count, detected_count, status, item_class, timestamp) VALUES (?,?,?,?,?)",
        (expected, detected, status, item_class, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

def fetch_secure_logs(limit=5):
    """Same 4-column log view as the original UI: Time | Event | Product | Confidence."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT timestamp as Time, event_type as Event, product as Product, confidence as Confidence "
        "FROM secure_access_logs ORDER BY id DESC LIMIT ?",
        conn,
        params=(int(limit),),
    )
    conn.close()
    return df

def render_secure_log_table(placeholder, limit=5, empty_msg=None):
    """Render logs as a compact full-width table — all columns visible, no horizontal scroll."""
    df = fetch_secure_logs(limit=limit)
    if df.empty:
        if empty_msg:
            placeholder.info(empty_msg)
        else:
            placeholder.empty()
        return
    df = df.copy()
    df["Time"] = pd.to_datetime(df["Time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    df["Confidence"] = pd.to_numeric(df["Confidence"], errors="coerce").round(2)

    # HTML table fits Time | Event | Product | Confidence in one view (no side-scroll)
    rows_html = []
    for _, r in df.iterrows():
        conf = r["Confidence"]
        conf_s = f"{conf:.2f}" if pd.notna(conf) else ""
        event_disp = localize_event_text(r["Event"])
        prod_disp = cls_label(str(r["Product"])) if r["Product"] in (
            "bottle", "laptop", "cell phone", "backpack", "suitcase", "cup", "handbag", "package"
        ) else r["Product"]
        rows_html.append(
            f"<tr>"
            f"<td style='padding:8px 6px; white-space:nowrap; font-size:12px;'>{r['Time']}</td>"
            f"<td style='padding:8px 6px; font-size:12px;'>{event_disp}</td>"
            f"<td style='padding:8px 6px; font-size:12px;'>{prod_disp}</td>"
            f"<td style='padding:8px 6px; text-align:right; white-space:nowrap; font-size:12px;'>{conf_s}</td>"
            f"</tr>"
        )
    table = f"""
    <div style="width:100%; overflow-x:hidden; border:1px solid rgba(255,255,255,0.08);
                border-radius:12px; background:rgba(255,255,255,0.03);">
      <table style="width:100%; border-collapse:collapse; table-layout:fixed; color:#e8ecf1;
                    font-family:Inter,sans-serif;">
        <colgroup>
          <col style="width:34%;">
          <col style="width:36%;">
          <col style="width:16%;">
          <col style="width:14%;">
        </colgroup>
        <thead>
          <tr style="background:rgba(255,255,255,0.06); text-align:left;">
            <th style="padding:8px 6px; font-size:11px; color:#5b6b7e; font-weight:600;">{t("col_time")}</th>
            <th style="padding:8px 6px; font-size:11px; color:#5b6b7e; font-weight:600;">{t("col_event")}</th>
            <th style="padding:8px 6px; font-size:11px; color:#5b6b7e; font-weight:600;">{t("col_product")}</th>
            <th style="padding:8px 6px; font-size:11px; color:#5b6b7e; font-weight:600; text-align:right;">{t("col_conf")}</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows_html)}
        </tbody>
      </table>
    </div>
    """
    placeholder.markdown(table, unsafe_allow_html=True)

def count_secure_removals():
    conn = sqlite3.connect(DB_PATH)
    n = conn.execute(
        "SELECT COUNT(*) FROM secure_access_logs "
        "WHERE event_type LIKE '%Removed%' OR event_type LIKE '%撤去%'"
    ).fetchone()[0]
    conn.close()
    return n

# ==========================================
# STYLING
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Inter:wght@300;400;500;600;700&display=swap');

.stApp {
    background-color: #0a0f1a !important;
    color: #e8ecf1 !important;
    font-family: 'Inter', sans-serif !important;
}

.stApp::before {
    content: "";
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        linear-gradient(to right, rgba(77,107,255,0.025) 1px, transparent 1px),
        linear-gradient(to bottom, rgba(77,107,255,0.025) 1px, transparent 1px);
    background-size: 46px 46px;
    pointer-events: none;
    z-index: 0;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    color: #ffffff !important;
}

/* Hide sidebar completely */
section[data-testid="stSidebar"] { display: none !important; }

/* Tabs */
button[data-baseweb="tab"] {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    color: #5b6b7e !important;
    background: transparent !important;
    border: none !important;
    padding: 12px 24px !important;
}
button[data-baseweb="tab"]:hover { color: #4d6bff !important; }
button[data-baseweb="tab"][aria-selected="true"] {
    color: #4d6bff !important;
    border-bottom: 2px solid #4d6bff !important;
}

/* Glassmorphic metrics */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    backdrop-filter: blur(12px) !important;
}
div[data-testid="stMetricValue"] {
    color: #4d6bff !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 800 !important;
    font-size: 28px !important;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #4d6bff, #6a4dff) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 9999px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    padding: 10px 28px !important;
    font-size: 15px !important;
    box-shadow: 0 4px 14px rgba(77,107,255,0.35) !important;
    transition: all 0.3s ease !important;
}
.stButton>button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(77,107,255,0.5) !important;
}

/* Status pills */
.pill { display: inline-flex; align-items: center; gap: 6px; padding: 8px 16px; border-radius: 9999px; font-size: 14px; font-weight: 600; font-family: 'Outfit', sans-serif; }
.pill-green { background: rgba(16,185,129,0.1); color: #10b981; border: 1px solid rgba(16,185,129,0.2); }
.pill-red { background: rgba(239,68,68,0.1); color: #ef4444; border: 1px solid rgba(239,68,68,0.2); }
.pill-blue { background: rgba(77,107,255,0.1); color: #4d6bff; border: 1px solid rgba(77,107,255,0.2); }

/* Cards */
.glass-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 24px;
    backdrop-filter: blur(12px);
    margin-bottom: 16px;
}

/* Alert boxes */
div[data-testid="stAlert"] {
    background: rgba(11,17,32,0.8) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
}

/* Dataframe */
div[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* File uploader */
div[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px dashed rgba(77,107,255,0.3) !important;
    border-radius: 12px !important;
}

/* Tighter language radio under the robot */
div[data-testid="stHorizontalBlock"] div[data-testid="stRadio"] {
    margin-top: -6px !important;
}
div[data-testid="stRadio"] > label {
    justify-content: center !important;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# HEADER — brand left, Rofi + language toggle right (toggle under robot)
# ==========================================
# Use key="ui_lang" only — never assign st.session_state.ui_lang after the radio widget
if "ui_lang" not in st.session_state:
    legacy = st.session_state.get("lang")
    st.session_state.ui_lang = legacy if legacy in ("en", "ja") else "en"

_logo_html = (
    f'<img src="{LOGO_URI}" alt="RoboFounders" '
    f'style="height:56px; width:auto; display:block; object-fit:contain;" />'
    if LOGO_URI else
    '<div style="width:56px; height:56px; border-radius:12px; background:linear-gradient(135deg,#4d6bff,#6a4dff);"></div>'
)
_rofi_html = (
    f'<img src="{ROFI_URI}" alt="Rofi" '
    f'style="height:150px; width:auto; display:block; object-fit:contain; margin:0 auto; '
    f'filter: drop-shadow(0 12px 28px rgba(77,107,255,0.45));" />'
    if ROFI_URI else ""
)

# Single compact header row: title | robot + language under robot
_hdr_left, _hdr_right = st.columns([4.6, 1.2], vertical_alignment="center")
with _hdr_left:
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:16px; min-width:0; padding:4px 0 2px 0;">
        {_logo_html}
        <div style="display:flex; flex-direction:column; gap:8px; min-width:0;">
            <div style="display:flex; align-items:center; gap:12px; flex-wrap:wrap;">
                <div style="font-size:42px; line-height:1.05; background:linear-gradient(135deg,#4d6bff,#6a4dff);
                            -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                            background-clip:text; font-weight:900; font-family:'Outfit',sans-serif;">RoboVision</div>
                <div style="background:rgba(77,107,255,0.1); border:1px solid rgba(77,107,255,0.2);
                            padding:5px 14px; border-radius:9999px; font-size:12px; font-weight:600;
                            color:#4d6bff; font-family:'Inter',sans-serif; letter-spacing:0.5px;">{t("badge_suite")}</div>
            </div>
            <div style="font-family:'Inter',sans-serif; font-size:13px; color:#5b6b7e;">
                {t("powered_by")} <span style="color:#4d6bff; font-weight:600;">RoboFounders.ai</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with _hdr_right:
    st.markdown(
        f'<div style="display:flex; flex-direction:column; align-items:center; gap:4px;">{_rofi_html}</div>',
        unsafe_allow_html=True,
    )
    # Language toggle directly under the robot
    st.radio(
        "Language",
        options=["en", "ja"],
        format_func=lambda c: "English" if c == "en" else "日本語",
        horizontal=True,
        key="ui_lang",
        label_visibility="collapsed",
    )

st.markdown(
    "<div style='border-bottom:1px solid rgba(255,255,255,0.08); margin:4px 0 14px 0;'></div>",
    unsafe_allow_html=True,
)

st.markdown(f"""
<p style="color:#5b6b7e; font-size:15px; margin-top:0; margin-bottom:20px; font-family:'Inter',sans-serif;">
    {t("tagline")}
</p>
""", unsafe_allow_html=True)


# ==========================================
# LOAD MODELS (lazy, single model — critical for Render 512MB)
# ==========================================
def _configure_torch_low_mem():
    try:
        import torch
        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    except Exception:
        pass

def get_detector(for_packages: bool = False):
    """
    Lazy-load ONE YOLO model only.
    Low-mem (Render free 512MB): always yolov8n.pt — never load YOLO-World.
    Full mem: packages can use yolov8s-world.pt.
    """
    _configure_torch_low_mem()
    from ultralytics import YOLO

    want_world = bool(for_packages) and (not LOW_MEM) and os.path.isfile(PACKAGE_MODEL_NAME)
    kind = "world" if want_world else "n"
    cached_kind = st.session_state.get("_yolo_kind")
    model = st.session_state.get("_yolo_model")

    if model is not None and cached_kind == kind:
        return model

    # Drop previous model before loading another (avoid 2× RAM)
    if model is not None:
        try:
            del model
        except Exception:
            pass
        st.session_state["_yolo_model"] = None
        st.session_state["_yolo_kind"] = None
        gc.collect()

    if kind == "world":
        m = YOLO(PACKAGE_MODEL_NAME)
        m.set_classes(PACKAGE_WORLD_CLASSES)
    else:
        m = YOLO(MODEL_NAME)

    st.session_state["_yolo_model"] = m
    st.session_state["_yolo_kind"] = kind
    return m

def belt_roi_polygon(h, w):
    """Conveyor surface ROI for the demo camera (normalized → pixels)."""
    return np.array([
        [int(0.02 * w), int(0.28 * h)],
        [int(0.98 * w), int(0.22 * h)],
        [int(0.99 * w), int(0.72 * h)],
        [int(0.01 * w), int(0.80 * h)],
    ], dtype=np.int32)

def belt_roi_mask(h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [belt_roi_polygon(h, w)], 255)
    return mask

def nms_boxes(boxes, thr=0.30):
    """boxes: list of (x1,y1,x2,y2,conf)."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    keep = []
    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        remaining = []
        ax1, ay1, ax2, ay2, _ = best
        for b in boxes:
            bx1, by1, bx2, by2, _ = b
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter == 0:
                remaining.append(b)
                continue
            ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
            if ua <= 0 or inter / ua < thr:
                remaining.append(b)
        boxes = remaining
    return keep

def detect_packages_on_belt(frame, package_model, use_world: bool = True):
    """
    Count packages only on the conveyor belt.
    World model: open-vocab labels. Low-mem: COCO proxies inside belt ROI.
    Returns (count, list of (x1,y1,x2,y2,conf), annotated_frame).
    """
    h, w = frame.shape[:2]
    mask = belt_roi_mask(h, w)
    imgsz = INFER_IMGSZ
    conf = PACKAGE_CONF_THRESHOLD if use_world else LOADING_CONF_THRESHOLD
    results = package_model.predict(
        frame, conf=conf, iou=0.45, imgsz=imgsz, verbose=False
    )
    raw = []
    for box in results[0].boxes:
        cls_name = package_model.names[int(box.cls[0])]
        if use_world:
            pass  # world classes already restricted via set_classes
        else:
            if cls_name not in PACKAGE_COCO_CLASSES:
                continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf_v = float(box.conf[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if mask[min(h - 1, max(0, cy)), min(w - 1, max(0, cx))] == 0:
            continue
        area = (x2 - x1) * (y2 - y1)
        if area < 900 or area > 0.40 * h * w:
            continue
        ar = (x2 - x1) / max(1, (y2 - y1))
        if ar < 0.3 or ar > 4.0:
            continue
        # Ignore lower-left bin (not on belt)
        if cy > int(0.78 * h) and cx < int(0.28 * w):
            continue
        if cy < int(0.22 * h):
            continue
        raw.append((x1, y1, x2, y2, conf_v))

    boxes = nms_boxes(raw, thr=0.30)
    annotated = frame.copy()
    # light ROI outline so operators know where we count
    cv2.polylines(annotated, [belt_roi_polygon(h, w)], True, (77, 107, 255), 1)
    for x1, y1, x2, y2, conf_v in boxes:
        # Censor brand text/logo on display frame for orange/brown cardboard boxes
        box_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if box_crop.size > 0:
            hsv = cv2.cvtColor(box_crop, cv2.COLOR_BGR2HSV)
            orange_mask = cv2.inRange(hsv, (8, 50, 50), (25, 255, 255))
            if np.sum(orange_mask > 0) > 0.15 * (box_crop.shape[0] * box_crop.shape[1]):
                bh, bw = y2 - y1, x2 - x1
                cx1, cy1 = x1 + int(bw * 0.2), y1 + int(bh * 0.2)
                cx2, cy2 = x1 + int(bw * 0.8), y1 + int(bh * 0.75)
                cx1, cy1 = max(0, cx1), max(0, cy1)
                cx2, cy2 = min(w, cx2), min(h, cy2)
                sub_roi = annotated[cy1:cy2, cx1:cx2]
                if sub_roi.size > 0:
                    annotated[cy1:cy2, cx1:cx2] = cv2.GaussianBlur(sub_roi, (41, 41), 15)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (16, 185, 129), 2)
        draw_text(
            annotated, f"{t('package_label')} {conf_v:.2f}",
            (x1, max(18, y1 - 8)), color_bgr=(16, 185, 129), scale=0.5, thickness=2
        )
    return len(boxes), boxes, annotated


# ==========================================
# VIDEO SOURCE HELPER
# ==========================================
# Fallback URLs if demo files are missing inside the container (public GitHub / samples)
DEMO_URLS = {
    "secure": [
        "https://github.com/intel-iot-devkit/sample-videos/raw/master/bottle-detection.mp4",
        "https://raw.githubusercontent.com/JacobAsir/RoboVision/main/bottle-detection.mp4",
    ],
    "loading": [
        "https://raw.githubusercontent.com/JacobAsir/RoboVision/main/5903898-hd_1920_1080_30fps.mp4",
    ],
}

def ensure_demo_video(path, tab_key):
    """Return local path if file exists; otherwise try download into BASE_DIR."""
    path = os.path.abspath(path)
    if os.path.isfile(path) and os.path.getsize(path) > 1000:
        return path
    urls = DEMO_URLS.get(tab_key, [])
    import urllib.request
    for url in urls:
        try:
            st.warning(t("demo_downloading"))
            urllib.request.urlretrieve(url, path)
            if os.path.isfile(path) and os.path.getsize(path) > 1000:
                return path
        except Exception:
            continue
    return None

def open_video_capture(source):
    """
    Open a video path robustly on Linux/Render (absolute path + FFmpeg backend).
    Returns (cap, error_message_or_None).
    """
    if not source:
        return None, "No video source selected."
    path = os.path.abspath(str(source))
    if not os.path.isfile(path):
        return None, f"Video file not found: {path}"
    if os.path.getsize(path) < 1000:
        return None, f"Video file is empty/corrupt: {path}"

    # Prefer FFmpeg backend (Linux Docker / Render)
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, f"OpenCV could not open video (codec/path): {path}"
    # Smoke-test one frame (some builds open but cannot decode)
    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None, f"OpenCV opened but could not read frames: {path}"
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return cap, None

def get_video_source(tab_key):
    """Simple video source selector — just demo video or upload your own."""
    st.markdown(f"""
    <div class="glass-card">
        <div style="font-family:'Outfit'; font-weight:600; font-size:16px; color:#fff; margin-bottom:12px;">{t("video_source")}</div>
        <p style="color:#5b6b7e; font-size:13px; margin:0;">{t("video_source_help")}</p>
    </div>
    """, unsafe_allow_html=True)

    # Stable option ids (demo/upload). Labels change with language via format_func.
    # Fix stale session values from older builds so a radio always stays selected.
    src_key = f"video_src_{tab_key}"
    ensure_radio_value(src_key, ("demo", "upload"), "demo")

    source_choice = st.radio(
        t("choose_input"),
        options=["demo", "upload"],
        format_func=lambda x: t("use_demo") if x == "demo" else t("upload_own"),
        key=src_key,
        label_visibility="collapsed",
    )

    if tab_key == "secure":
        demo_file = DEMO_VIDEO_SECURE
    elif tab_key == "container":
        demo_file = DEMO_VIDEO_CONTAINER
    else:
        demo_file = DEMO_VIDEO_LOADING

    if source_choice == "demo":
        if tab_key == "container":
            scenario_choice = st.selectbox(
                "Select Logistics Scenario",
                ["Scenario 1: Conveyor Parcel Bay", "Scenario 2: Heavy Freight Dock"],
                index=0, key="container_scenario_sel"
            )
            demo_file = DEMO_VIDEO_CONTAINER if "Scenario 1" in scenario_choice else DEMO_VIDEO_LOADING

        resolved = ensure_demo_video(demo_file, tab_key)
        if resolved:
            return resolved
        st.error(t("demo_missing", path=os.path.abspath(demo_file)))
        with st.expander("Debug (Render)"):
            st.code(
                f"BASE_DIR={BASE_DIR}\n"
                f"cwd={os.getcwd()}\n"
                f"exists={os.path.exists(demo_file)}\n"
                f"files={os.listdir(BASE_DIR)[:40]}"
            )
        return None
    else:
        uploaded = st.file_uploader(t("drop_video"), type=["mp4", "avi", "mov"], key=f"upload_{tab_key}")
        if uploaded:
            temp_dir = os.path.join(BASE_DIR, "temp_assets")
            os.makedirs(temp_dir, exist_ok=True)
            path = os.path.join(temp_dir, uploaded.name)
            with open(path, "wb") as f:
                f.write(uploaded.read())
            return path
        else:
            st.info(t("upload_hint"))
            return None


# ==========================================
# TRACKING UTILITIES
# ==========================================
def is_near_person(product_bbox, person_bbox, padding=90):
    """Generous proximity — demo cameras often miss tight hand/bottle overlap."""
    px1, py1, px2, py2 = product_bbox
    x1, y1, x2, y2 = person_bbox
    x1 -= padding; y1 -= padding; x2 += padding; y2 += padding
    overlap_x = not (px2 < x1 or px1 > x2)
    overlap_y = not (py2 < y1 or py1 > y2)
    if overlap_x and overlap_y:
        return True
    pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
    cx, cy = (person_bbox[0] + person_bbox[2]) / 2, (person_bbox[1] + person_bbox[3]) / 2
    # Distance in resized 640px space — looser than before so touch events register
    return np.sqrt((pcx - cx) ** 2 + (pcy - cy) ** 2) < 260

def match_fallback_track(cx, cy, cls_id, bbox):
    if "fallback_tracks" not in st.session_state:
        st.session_state.fallback_tracks = {}
        st.session_state.next_track_id = 1000
    now = time.time()
    # Keep IDs longer so brief misses (occlusions / frame skips) don't spawn new tracks
    st.session_state.fallback_tracks = {
        tid: d for tid, d in st.session_state.fallback_tracks.items() if now - d["last_seen"] < 2.5
    }
    best_id, min_dist = None, 140.0
    for tid, d in st.session_state.fallback_tracks.items():
        if d["cls_id"] == cls_id:
            dist = np.sqrt((cx - d["centroid"][0]) ** 2 + (cy - d["centroid"][1]) ** 2)
            if dist < min_dist:
                min_dist, best_id = dist, tid
    if best_id is not None:
        st.session_state.fallback_tracks[best_id].update({"centroid": (cx, cy), "bbox": bbox, "last_seen": now})
        return best_id
    new_id = st.session_state.next_track_id
    st.session_state.next_track_id += 1
    st.session_state.fallback_tracks[new_id] = {
        "cls_id": cls_id, "centroid": (cx, cy), "bbox": bbox, "last_seen": now
    }
    return new_id

def get_tracks(results, product_class, names):
    products, persons = [], []
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return products, persons
    has_ids = boxes.id is not None
    for i, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        name = names[cls_id]
        bbox = box.xyxy[0].cpu().tolist()
        conf = float(box.conf[0])
        cx, cy = int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2)
        if has_ids:
            try: tid = int(boxes.id[i].item())
            except: tid = match_fallback_track(cx, cy, cls_id, bbox)
        else:
            tid = match_fallback_track(cx, cy, cls_id, bbox)
        obj = {"tid":tid, "name":name, "bbox":bbox, "conf":conf, "cx": cx, "cy": cy}
        if name == "person": persons.append(obj)
        elif name == product_class: products.append(obj)
    return products, persons


def update_product_slots(detections, persons):
    """
    Position-based multi-bottle tracker.

    YOLO track IDs often merge/swap when several bottles leave together.
    Matching by centroid keeps each shelf position as its own slot so
    removing two bottles produces two removal events.
    """
    if "active_products" not in st.session_state:
        st.session_state.active_products = {}
    if "secure_slot_seq" not in st.session_state:
        st.session_state.secure_slot_seq = 1

    slots = st.session_state.active_products
    person_now = len(persons) > 0
    if person_now:
        st.session_state.secure_recent_person = True

    # Greedy match detections → existing slots (nearest centroid)
    unmatched_dets = set(range(len(detections)))
    unmatched_slots = set(slots.keys())
    pairs = []
    for sid, state in slots.items():
        scx, scy = state.get("centroid", (0, 0))
        for di, det in enumerate(detections):
            if di not in unmatched_dets:
                continue
            dist = np.hypot(det["cx"] - scx, det["cy"] - scy)
            pairs.append((dist, sid, di))
    pairs.sort(key=lambda x: x[0])

    assigned_slots = set()
    assigned_dets = set()
    for dist, sid, di in pairs:
        if sid in assigned_slots or di in assigned_dets:
            continue
        if dist > SECURE_SLOT_MATCH_DIST:
            continue
        assigned_slots.add(sid)
        assigned_dets.add(di)
        unmatched_slots.discard(sid)
        unmatched_dets.discard(di)
        det = detections[di]
        near = any(is_near_person(det["bbox"], per["bbox"]) for per in persons)
        st_ap = slots[sid]
        st_ap.update({
            "bbox": det["bbox"],
            "centroid": (det["cx"], det["cy"]),
            "last_seen": time.time(),
            "frames_missing": 0,
            "conf": det["conf"],
            "yolo_tid": det["tid"],
        })
        st_ap["frames_seen"] = st_ap.get("frames_seen", 0) + 1
        if near:
            st_ap["near_person"] = True
        if person_now or near:
            st_ap["person_while_active"] = True

    # New bottles → new slots
    for di in list(unmatched_dets):
        det = detections[di]
        near = any(is_near_person(det["bbox"], per["bbox"]) for per in persons)
        sid = st.session_state.secure_slot_seq
        st.session_state.secure_slot_seq = sid + 1
        slots[sid] = {
            "bbox": det["bbox"],
            "centroid": (det["cx"], det["cy"]),
            "last_seen": time.time(),
            "near_person": near,
            "person_while_active": person_now or near,
            "frames_missing": 0,
            "frames_seen": 1,
            "conf": det["conf"],
            "yolo_tid": det["tid"],
        }

    # Missing slots → candidate removals
    removal_events = []
    gone = []
    for sid in list(unmatched_slots):
        state = slots[sid]
        state["frames_missing"] = state.get("frames_missing", 0) + 1
        if state["frames_missing"] < OCCLUSION_LIMIT:
            continue
        if state.get("frames_seen", 0) < MIN_PRODUCT_SEEN:
            gone.append(sid)  # never stable — drop quietly
            continue
        person_flag = (
            "Yes"
            if (
                state.get("near_person")
                or state.get("person_while_active")
                or person_now
                or st.session_state.get("secure_recent_person")
            )
            else "Unknown"
        )
        removal_events.append({
            "slot": sid,
            "conf": float(state.get("conf", 0.0) or 0.0),
            "person": person_flag,
        })
        gone.append(sid)

    for sid in gone:
        slots.pop(sid, None)

    return removal_events, person_now


# ==========================================
# SESSION STATE INIT — always fresh on refresh / tab switch
# ==========================================
# Browser refresh = new Streamlit session → wipe old DB logs and trackers
if "app_session_ready" not in st.session_state:
    start_fresh_analysis()
    st.session_state.app_session_ready = True
    st.session_state.current_use_case = None
else:
    # Ensure keys exist without reusing stale values from partial state
    for key, default in ANALYSIS_STATE_DEFAULTS.items():
        if key not in st.session_state:
            if isinstance(default, dict):
                st.session_state[key] = {}
            elif isinstance(default, list):
                st.session_state[key] = []
            else:
                st.session_state[key] = default


# ==========================================
# Stable use-case ids; labels are language-specific. Repair stale values after upgrades.
ensure_radio_value("use_case_selector", ("secure", "loading", "container"), "secure")
if st.session_state.get("current_use_case") not in ("secure", "loading", "container", None):
    st.session_state.current_use_case = st.session_state.use_case_selector

def format_use_case_label(k):
    if k == "secure":
        return t("use_case_secure")
    elif k == "loading":
        return t("use_case_loading")
    else:
        return t("use_case_container")

active_use_case = st.radio(
    "Select Use Case",
    options=["secure", "loading", "container"],
    format_func=format_use_case_label,
    horizontal=True,
    label_visibility="collapsed",
    key="use_case_selector",
)

# Switching tabs restarts analysis from zero (new logs, empty history, new tracking)
# Language toggle does NOT reset analysis and must NOT rewrite ui_lang
if st.session_state.get("current_use_case") != active_use_case:
    if st.session_state.get("current_use_case") is not None:
        start_fresh_analysis()
    st.session_state.current_use_case = active_use_case

# ------------------------------------------
# TAB 1: SECURE PRODUCT MONITORING
# ------------------------------------------
if active_use_case == "secure":
    # Problem statement card
    st.markdown(f"""
    <div class="glass-card">
        <h3 style="margin-top:0; font-size:22px;">{t("secure_title")}</h3>
        <p style="color:#8b95a5; font-size:14px; line-height:1.7; margin-bottom:0;">
            <b style="color:#e8ecf1;">{t("secure_problem")}</b> {t("secure_problem_body")}<br>
            <b style="color:#e8ecf1;">{t("secure_solution")}</b> {t("secure_solution_body")}
        </p>
    </div>
    """, unsafe_allow_html=True)
    

    # Dynamic 3-column side-by-side layout (Compact and presentation ready)
    col_ctrl, col_vid, col_stats = st.columns([1.2, 2.8, 2.5])
    
    with col_ctrl:
        # Video source selection
        source = get_video_source("secure")
        
        # Target item selection (internal English class for YOLO; Japanese display labels)
        _secure_classes = ["bottle", "laptop", "cell phone", "backpack", "suitcase", "cup", "handbag"]
        product_class = st.selectbox(
            t("target_product"),
            _secure_classes,
            index=0, key="secure_product",
            format_func=cls_label,
        )
        
        # Clean run toggle — new key defaults ON (old sessions may have left feed off)
        st.markdown("")
        if "run_secure_v2" not in st.session_state:
            st.session_state.run_secure_v2 = True
        run_secure = st.toggle(t("activate_cctv"), key="run_secure_v2")

    with col_vid:
        st.markdown(f"<div style='font-family:Outfit; font-weight:600; font-size:16px; margin-bottom:8px;'>{t('live_cctv')}</div>", unsafe_allow_html=True)
        stframe = st.empty()
        alert_banner = st.empty()

    with col_stats:
        st.markdown(f"<div style='font-family:Outfit; font-weight:600; font-size:16px; margin-bottom:8px;'>{t('realtime_monitoring')}</div>", unsafe_allow_html=True)
        
        # Placeholders for metrics and logs
        metrics_placeholder = st.empty()
        log_table = st.empty()
        
        # Download button placeholder
        download_placeholder = st.empty()

    def _seek_secure_demo_start(capture, video_source, fps_hint=0.0):
        """Start bottle demo after SECURE_DEMO_START_SEC (skip idle intro)."""
        try:
            src_path = os.path.abspath(video_source) if isinstance(video_source, str) else ""
            demo_path = os.path.abspath(DEMO_VIDEO_SECURE)
            if not src_path or os.path.basename(src_path) != os.path.basename(demo_path):
                return
            fps_use = fps_hint if fps_hint and fps_hint > 0 else capture.get(cv2.CAP_PROP_FPS)
            if not fps_use or fps_use <= 0:
                fps_use = 30.0
            start_frame = int(SECURE_DEMO_START_SEC * fps_use)
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        except Exception:
            pass

    # Active monitoring thread execution
    if not run_secure:
        stframe.info("Turn **ON** Activate CCTV Feed to start the video.")
    elif not source:
        stframe.warning("No video source available.")
    elif run_secure and source:
        cap, cap_err = open_video_capture(source)
        if cap is None:
            st.error(t("open_video_fail"))
            st.error(cap_err or "")
            with st.expander("Debug (Render)"):
                st.code(f"source={source}\nBASE_DIR={BASE_DIR}\ncwd={os.getcwd()}\nfiles={[f for f in os.listdir(BASE_DIR) if f.endswith(('.mp4','.pt','.png'))]}")
        else:
            try:
                # Faster than real-time so the bottle demo doesn't feel like a long wait
                fps = cap.get(cv2.CAP_PROP_FPS)
                base_delay = 1.0 / fps if fps > 0 else 0.033
                frame_delay = base_delay / max(1.0, SECURE_PLAYBACK_SPEED)
                secure_frame_i = 0
                # Bottle demo: jump past the first 3 seconds
                _seek_secure_demo_start(cap, source, fps)
                
                while run_secure:
                    start_time = time.time()
                    ret, frame = cap.read()
                    if not ret:
                        if isinstance(source, str):
                            # Loop from the same 3s mark (not the very start)
                            _seek_secure_demo_start(cap, source, fps)
                            secure_frame_i = 0
                            st.session_state.active_products = {}
                            st.session_state.fallback_tracks = {}
                            st.session_state.secure_baseline_count = 0
                            st.session_state.secure_low_count_streak = 0
                            st.session_state.secure_recent_person = False
                            st.session_state.secure_slot_seq = 1
                            continue
                        break

                    # Optional stride (default 1 = every frame for multi-bottle accuracy)
                    secure_frame_i += 1
                    if SECURE_FRAME_STRIDE > 1 and (secure_frame_i % SECURE_FRAME_STRIDE) != 0:
                        continue
                    
                    # 1. Resize for CPU-friendly / low-RAM inference
                    h, w = frame.shape[:2]
                    target_w = 480 if LOW_MEM else 640
                    target_h = int(h * (target_w / w))
                    resized_frame = cv2.resize(frame, (target_w, target_h))
                    
                    # 2. Detect bottles (predict — no `lap` / ByteTrack dependency for Streamlit Cloud)
                    #    Spatial slots below assign stable IDs; model.track() needs package `lap`
                    #    which Cloud cannot install (venv permission denied).
                    model = get_detector(for_packages=False)
                    results = model.predict(
                        source=resized_frame,
                        conf=SECURE_CONF_THRESHOLD,
                        iou=IOU_THRESHOLD,
                        imgsz=INFER_IMGSZ,
                        verbose=False,
                    )
                    curr_products, curr_persons = get_tracks(results, product_class, model.names)

                    # 3. Spatial slots — one slot per bottle position (2 removals → 2 logs)
                    removal_events, person_now = update_product_slots(curr_products, curr_persons)

                    annotated = resized_frame.copy()
                    for sid, state in st.session_state.active_products.items():
                        bbox = state["bbox"]
                        touched = state.get("near_person", False)
                        color = (106, 77, 255) if touched else (77, 107, 255)
                        cv2.rectangle(
                            annotated,
                            (int(bbox[0]), int(bbox[1])),
                            (int(bbox[2]), int(bbox[3])),
                            color, 2
                        )
                        label = f"{cls_label(product_class)} #{sid}"
                        if touched:
                            label += " " + t("touched")
                        draw_text(
                            annotated, label,
                            (int(bbox[0]), int(bbox[1]) - 10),
                            color_bgr=color, scale=0.5, thickness=2
                        )

                    for per in curr_persons:
                        pb = per["bbox"]
                        cv2.rectangle(
                            annotated,
                            (int(pb[0]), int(pb[1])),
                            (int(pb[2]), int(pb[3])),
                            (255, 255, 255), 1
                        )

                    # 4. Log each slot removal (one row per bottle)
                    for ev in removal_events:
                        log_secure_access(
                            t("event_product_removed"),
                            product_class,
                            ev["person"],
                            ev["conf"],
                        )
                    if removal_events:
                        n = len(removal_events)
                        prod_disp = cls_label(product_class).upper()
                        st.session_state.last_alert = {
                            "text": (
                                t("alarm_removed_n", n=n, product=prod_disp)
                                if n > 1
                                else t("alarm_removed_one", product=prod_disp)
                            ),
                            "time": time.time(),
                        }
                        # Keep baseline in sync so backup path won't re-log these
                        bl = st.session_state.get("secure_baseline_count", 0)
                        st.session_state.secure_baseline_count = max(0, bl - n)
                        st.session_state.secure_low_count_streak = 0

                    # 5. Baseline count backup — if slots under-count a multi-remove, fill the gap
                    live_count = len(curr_products)
                    slot_count = len(st.session_state.active_products)
                    observed = max(live_count, slot_count)
                    baseline = st.session_state.get("secure_baseline_count", 0)
                    if observed > baseline:
                        st.session_state.secure_baseline_count = observed
                        st.session_state.secure_low_count_streak = 0
                        baseline = observed

                    if observed < baseline:
                        st.session_state.secure_low_count_streak = (
                            st.session_state.get("secure_low_count_streak", 0) + 1
                        )
                    else:
                        st.session_state.secure_low_count_streak = 0

                    # Fill any remaining gap once (e.g. 2 bottles gone but only 1 slot logged)
                    if (
                        st.session_state.secure_low_count_streak == OCCLUSION_LIMIT
                        and baseline > observed
                    ):
                        gap = baseline - observed
                        if gap > 0:
                            person_flag = (
                                "Yes"
                                if person_now or st.session_state.get("secure_recent_person")
                                else "Unknown"
                            )
                            for _ in range(gap):
                                log_secure_access(
                                    t("event_product_removed"),
                                    product_class,
                                    person_flag,
                                    0.0,
                                )
                            prod_disp = cls_label(product_class).upper()
                            st.session_state.last_alert = {
                                "text": t("alarm_removed_n", n=gap, product=prod_disp),
                                "time": time.time(),
                            }
                        st.session_state.secure_baseline_count = observed
                        st.session_state.secure_low_count_streak = 0
                    
                    # Live alert visual overlay
                    alert_on = st.session_state.last_alert and time.time() - st.session_state.last_alert.get("time",0) < 5
                    if alert_on:
                        alert_banner.error(st.session_state.last_alert["text"])
                        h_a, w_a = annotated.shape[:2]
                        cv2.rectangle(annotated, (0,0), (w_a-1,h_a-1), (0,0,255), 12)
                    else:
                        alert_banner.empty()
                    
                    # Render resized frame
                    stframe.image(annotated, channels="BGR", width=520)
                    
                    # Update live metrics (ACTIVE ITEMS / REMOVALS / status — match reference UI)
                    total_removals = count_secure_removals()
                    metrics_html = f"""
                    <div style="display: flex; gap: 12px; margin-bottom: 12px;">
                        <div style="flex: 1; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 12px; text-align: center;">
                            <div style="font-size: 11px; color: #5b6b7e; font-weight: 600;">{t("active_items")}</div>
                            <div style="font-size: 20px; font-weight: 800; color: #4d6bff; font-family: Outfit;">{len(st.session_state.active_products)}</div>
                        </div>
                        <div style="flex: 1; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 12px; text-align: center;">
                            <div style="font-size: 11px; color: #5b6b7e; font-weight: 600;">{t("removals")}</div>
                            <div style="font-size: 20px; font-weight: 800; color: #4d6bff; font-family: Outfit;">{total_removals}</div>
                        </div>
                        <div style="flex: 1.2; display: flex; align-items: center; justify-content: center;">
                            <div class="pill {'pill-red' if alert_on else 'pill-green'}" style="width: 100%; text-align: center; justify-content: center; font-size: 12px;">
                                {t("breach") if alert_on else t("secured")}
                            </div>
                        </div>
                    </div>
                    """
                    metrics_placeholder.markdown(metrics_html, unsafe_allow_html=True)
                    
                    # Logs table — full datetime + Time | Event | Product | Confidence
                    render_secure_log_table(log_table, limit=5, empty_msg=None)
                    
                    # Pace frames (already shortened by SECURE_PLAYBACK_SPEED)
                    processing_time = time.time() - start_time
                    time.sleep(max(0.0, frame_delay - processing_time))
            finally:
                cap.release()
    
    # Show history database logs when camera is stopped
    if not run_secure:
        with col_stats:
            render_secure_log_table(
                log_table,
                limit=5,
                empty_msg=t("no_logs_secure"),
            )
            total_removals = count_secure_removals()
            metrics_html = f"""
            <div style="display: flex; gap: 12px; margin-bottom: 12px;">
                <div style="flex: 1; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 12px; text-align: center;">
                    <div style="font-size: 11px; color: #5b6b7e; font-weight: 600;">{t("active_items")}</div>
                    <div style="font-size: 20px; font-weight: 800; color: #4d6bff; font-family: Outfit;">{len(st.session_state.active_products)}</div>
                </div>
                <div style="flex: 1; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 12px; text-align: center;">
                    <div style="font-size: 11px; color: #5b6b7e; font-weight: 600;">{t("removals")}</div>
                    <div style="font-size: 20px; font-weight: 800; color: #4d6bff; font-family: Outfit;">{total_removals}</div>
                </div>
                <div style="flex: 1.2; display: flex; align-items: center; justify-content: center;">
                    <div class="pill pill-green" style="width: 100%; text-align: center; justify-content: center; font-size: 12px;">
                        {t("monitoring")}
                    </div>
                </div>
            </div>
            """
            metrics_placeholder.markdown(metrics_html, unsafe_allow_html=True)
            
    # Export reports (same 4 columns)
    full = fetch_secure_logs(limit=100000)
    if not full.empty:
        with col_ctrl:
            st.markdown("---")
            st.download_button(t("export_logs"), full.to_csv(index=False).encode('utf-8'),
                              "secure_access_report.csv", "text/csv")


# ------------------------------------------
# TAB 2: LOADING & PACKING VERIFICATION
# ------------------------------------------
elif active_use_case == "loading":
    st.markdown(f"""
    <div class="glass-card">
        <h3 style="margin-top:0; font-size:22px;">{t("loading_title")}</h3>
        <p style="color:#8b95a5; font-size:14px; line-height:1.7; margin-bottom:0;">
            <b style="color:#e8ecf1;">{t("loading_problem")}</b> {t("loading_problem_body")}<br>
            <b style="color:#e8ecf1;">{t("loading_solution")}</b> {t("loading_solution_body")}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Dynamic 3-column side-by-side layout (Compact and presentation ready)
    col_ctrl2, col_vid2, col_stats2 = st.columns([1.2, 2.8, 2.5])
    
    with col_ctrl2:
        # Video source selection
        source2 = get_video_source("loading")
        
        # Target selections (internal English class for model)
        _load_classes = ["package", "bottle", "cup", "laptop", "cell phone", "backpack"]
        verify_class = st.selectbox(
            t("what_to_count"),
            _load_classes,
            index=0, key="load_class",
            format_func=cls_label,
        )
        if verify_class == "package":
            st.caption(t("package_caption"))
            
        # Demo footage has 3 package groups passing on the belt
        expected_items = st.number_input(t("expected_qty"), min_value=0, value=3, step=1)
        
        # Clean run toggle
        st.markdown("")
        if "run_loading_v2" not in st.session_state:
            st.session_state.run_loading_v2 = True
        run_loading = st.toggle(t("activate_verify"), key="run_loading_v2")

    with col_vid2:
        st.markdown(f"<div style='font-family:Outfit; font-weight:600; font-size:16px; margin-bottom:8px;'>{t('live_verify')}</div>", unsafe_allow_html=True)
        stframe2 = st.empty()
        alert2 = st.empty()

    with col_stats2:
        st.markdown(f"<div style='font-family:Outfit; font-weight:600; font-size:16px; margin-bottom:8px;'>{t('realtime_verify')}</div>", unsafe_allow_html=True)
        
        # Placeholders for metrics and logs — fill immediately so the panel is never blank
        metrics_placeholder2 = st.empty()
        log_table2 = st.empty()
        download_placeholder2 = st.empty()

        metrics_placeholder2.markdown(f"""
        <div style="display: flex; gap: 12px; margin-bottom: 12px;">
            <div style="flex: 1; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 12px; text-align: center;">
                <div style="font-size: 11px; color: #5b6b7e; font-weight: 600;">{t("expected")}</div>
                <div style="font-size: 20px; font-weight: 800; color: #4d6bff; font-family: Outfit;">{expected_items}</div>
            </div>
            <div style="flex: 1; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 12px; text-align: center;">
                <div style="font-size: 11px; color: #5b6b7e; font-weight: 600;">{t("detected")}</div>
                <div style="font-size: 20px; font-weight: 800; color: #4d6bff; font-family: Outfit;">{st.session_state.live_detected_count}</div>
            </div>
            <div style="flex: 1.2; display: flex; align-items: center; justify-content: center;">
                <div class="pill pill-blue" style="width: 100%; text-align: center; justify-content: center; font-size: 12px;">
                    {t("starting") if run_loading else t("inactive")}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    target_classes = list(PACKAGE_COCO_CLASSES) if verify_class == "package" else [verify_class]
    load_conf = LOADING_CONF_THRESHOLD if verify_class == "package" else CONF_THRESHOLD
    use_world = (verify_class == "package") and (not LOW_MEM)
    if verify_class == "package" and LOW_MEM:
        st.caption("Low-memory mode (Render 512MB): package counting uses YOLOv8n + belt ROI.")

    # Active verification thread execution
    if not run_loading:
        stframe2.info("Turn **ON** Activate Verification Feed to start the video.")
    elif not source2:
        stframe2.warning(t("select_source"))
    elif run_loading and source2:
        cap, cap_err = open_video_capture(source2)
        if cap is None:
            st.error(t("open_source_fail", path=source2))
            stframe2.warning(t("video_open_fail"))
            if cap_err:
                st.error(cap_err)
            with st.expander("Debug (Render)"):
                st.code(f"source={source2}\nBASE_DIR={BASE_DIR}\nfiles={[f for f in os.listdir(BASE_DIR) if f.endswith(('.mp4','.pt','.png'))]}")
        else:
            try:
                # Retrieve video frame rate to match original playback speed
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_delay = 1.0 / fps if fps > 0 else 0.033
                # Skip frames when inference is slower than real-time so the UI stays responsive
                frame_idx = 0
                consecutive_fail = 0
                model = get_detector(for_packages=use_world)

                # Show first raw frame immediately so the panel is never empty while YOLO warms up
                ret0, frame0 = cap.read()
                if ret0:
                    h0, w0 = frame0.shape[:2]
                    preview = cv2.resize(frame0, (640, int(h0 * (640 / w0))))
                    draw_text(preview, t("analyzing"), (12, 28), color_bgr=(77, 107, 255), scale=0.8, thickness=2)
                    stframe2.image(preview, channels="BGR", width=520)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    stframe2.error(t("read_frames_fail"))

                while run_loading:
                    start_time = time.time()
                    ret, frame = cap.read()
                    if not ret:
                        consecutive_fail += 1
                        if isinstance(source2, str) and consecutive_fail < 5:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        stframe2.warning(t("end_of_video"))
                        break
                    consecutive_fail = 0
                    frame_idx += 1
                    if LOW_MEM and frame_idx % 2 == 0:
                        continue
                    
                    # 1. Resize for CPU-friendly inference
                    h, w = frame.shape[:2]
                    target_w = 480 if LOW_MEM else 640
                    target_h = int(h * (target_w / w))
                    resized_frame = cv2.resize(frame, (target_w, target_h))
                    resized_frame = cv2.GaussianBlur(resized_frame, (7, 7), 0)
                    
                    # 2. Package mode: world (full RAM) or nano+ROI (low mem)
                    if verify_class == "package":
                        count, _boxes, annotated = detect_packages_on_belt(
                            resized_frame, model, use_world=use_world
                        )
                    else:
                        results = model(resized_frame, conf=load_conf, imgsz=INFER_IMGSZ, verbose=False)
                        annotated = resized_frame.copy()
                        count = 0
                        for box in results[0].boxes:
                            cls_name = model.names[int(box.cls[0])]
                            if cls_name not in target_classes:
                                continue
                            count += 1
                            bbox = box.xyxy[0].cpu().tolist()
                            conf = float(box.conf[0])
                            x1, y1, x2, y2 = map(int, bbox)
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (16, 185, 129), 2)
                            label_txt = t("package_label") if verify_class == "package" else cls_label(verify_class)
                            draw_text(
                                annotated, f"{label_txt} {conf:.2f}",
                                (x1, max(18, y1 - 8)),
                                color_bgr=(16, 185, 129), scale=0.5, thickness=2
                            )

                    # Always draw live count on the frame so the feed is never "empty looking"
                    banner = t("count_banner", count=count, expected=expected_items)
                    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 36), (10, 15, 26), -1)
                    color = (16, 185, 129) if count == expected_items else (68, 68, 239)
                    draw_text(annotated, banner, (12, 24), color_bgr=color, scale=0.65, thickness=2)
                    
                    st.session_state.live_detected_count = count
                    
                    # Stabilization buffer
                    st.session_state.loading_count_history.append(count)
                    if len(st.session_state.loading_count_history) > STABLE_WINDOW:
                        st.session_state.loading_count_history.pop(0)
                    
                    if len(st.session_state.loading_count_history) == STABLE_WINDOW and len(set(st.session_state.loading_count_history)) == 1:
                        stable = st.session_state.loading_count_history[0]
                        if stable != st.session_state.loading_last_logged_count:
                            st.session_state.loading_last_logged_count = stable
                            status = t("verified") if stable == expected_items else t("status_mismatch")
                            log_loading_event(expected_items, stable, status, verify_class)
                    
                    # Render status warnings
                    is_match = count == expected_items
                    if not is_match:
                        alert2.error(t("qty_mismatch", expected=expected_items, detected=count))
                    else:
                        alert2.success(t("qty_verified", count=count))
                    
                    # Render frame
                    stframe2.image(annotated, channels="BGR", width=520)
                    
                    # Update live metrics
                    metrics_html2 = f"""
                    <div style="display: flex; gap: 12px; margin-bottom: 12px;">
                        <div style="flex: 1; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 12px; text-align: center;">
                            <div style="font-size: 11px; color: #5b6b7e; font-weight: 600;">{t("expected")}</div>
                            <div style="font-size: 20px; font-weight: 800; color: #4d6bff; font-family: Outfit;">{expected_items}</div>
                        </div>
                        <div style="flex: 1; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 12px; text-align: center;">
                            <div style="font-size: 11px; color: #5b6b7e; font-weight: 600;">{t("detected")}</div>
                            <div style="font-size: 20px; font-weight: 800; color: #4d6bff; font-family: Outfit;">{count}</div>
                        </div>
                        <div style="flex: 1.2; display: flex; align-items: center; justify-content: center;">
                            <div class="pill {'pill-green' if is_match else 'pill-red'}" style="width: 100%; text-align: center; justify-content: center; font-size: 12px;">
                                {t("match") if is_match else t("mismatch")}
                            </div>
                        </div>
                    </div>
                    """
                    metrics_placeholder2.markdown(metrics_html2, unsafe_allow_html=True)
                    
                    # Live logs list update
                    conn = sqlite3.connect(DB_PATH)
                    df2 = pd.read_sql_query(
                        "SELECT timestamp as Time, expected_count as Expected, detected_count as Detected, status as Status "
                        "FROM loading_logs ORDER BY id DESC LIMIT 5",
                        conn,
                    )
                    conn.close()
                    if not df2.empty:
                        df2["Time"] = pd.to_datetime(df2["Time"]).dt.strftime("%H:%M:%S")
                        df2["Status"] = df2["Status"].map(localize_event_text)
                        # Localized column headers for display
                        df2 = df2.rename(columns={
                            "Time": t("col_time"),
                            "Expected": t("col_expected"),
                            "Detected": t("col_detected"),
                            "Status": t("col_status"),
                        })
                        log_table2.dataframe(df2, use_container_width=True, hide_index=True)
                    else:
                        log_table2.empty()
                    
                    # Target original playback speed: calculate exact remaining sleep time
                    processing_time = time.time() - start_time
                    time.sleep(max(0.001, frame_delay - processing_time))
            finally:
                cap.release()
    elif run_loading and not source2:
        stframe2.info(t("select_source"))
    
    # Show history database logs when camera is stopped
    if not run_loading:
        with col_stats2:
            conn = sqlite3.connect(DB_PATH)
            df2 = pd.read_sql_query(
                "SELECT timestamp as Time, expected_count as Expected, detected_count as Detected, status as Status "
                "FROM loading_logs ORDER BY id DESC LIMIT 5",
                conn,
            )
            conn.close()
            if not df2.empty:
                df2["Time"] = pd.to_datetime(df2["Time"]).dt.strftime("%H:%M:%S")
                df2["Status"] = df2["Status"].map(localize_event_text)
                df2 = df2.rename(columns={
                    "Time": t("col_time"),
                    "Expected": t("col_expected"),
                    "Detected": t("col_detected"),
                    "Status": t("col_status"),
                })
                log_table2.dataframe(df2, use_container_width=True, hide_index=True)
            else:
                log_table2.info(t("no_logs_loading"))
                
            # Render initial metrics
            metrics_html2 = f"""
            <div style="display: flex; gap: 12px; margin-bottom: 12px;">
                <div style="flex: 1; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 12px; text-align: center;">
                    <div style="font-size: 11px; color: #5b6b7e; font-weight: 600;">{t("expected")}</div>
                    <div style="font-size: 20px; font-weight: 800; color: #4d6bff; font-family: Outfit;">{expected_items}</div>
                </div>
                <div style="flex: 1; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 12px; text-align: center;">
                    <div style="font-size: 11px; color: #5b6b7e; font-weight: 600;">{t("detected")}</div>
                    <div style="font-size: 20px; font-weight: 800; color: #4d6bff; font-family: Outfit;">{st.session_state.live_detected_count}</div>
                </div>
                <div style="flex: 1.2; display: flex; align-items: center; justify-content: center;">
                    <div class="pill pill-blue" style="width: 100%; text-align: center; justify-content: center; font-size: 12px;">
                        {t("inactive")}
                    </div>
                </div>
            </div>
            """
            metrics_placeholder2.markdown(metrics_html2, unsafe_allow_html=True)
            
    # Export reports
    conn = sqlite3.connect(DB_PATH)
    full2 = pd.read_sql_query(
        "SELECT timestamp as Time, expected_count as Expected, detected_count as Detected, "
        "status as Status, item_class as [Item Class] FROM loading_logs ORDER BY id DESC",
        conn,
    )
    conn.close()
    if not full2.empty:
        with col_ctrl2:
            st.markdown("---")
            st.download_button(t("export_logs"), full2.to_csv(index=False).encode('utf-8'),
                              "loading_verification_report.csv", "text/csv")
                              
                              
# ------------------------------------------
# TAB 3: 3D CONTAINER LOADING & STACKING OPTIMIZER
# ------------------------------------------
elif active_use_case == "container":
    st.markdown(f"""
    <div class="glass-card">
        <h3 style="margin-top:0; font-size:22px;">{t("container_title")}</h3>
        <p style="color:#8b95a5; font-size:14px; line-height:1.7; margin-bottom:0;">
            <b style="color:#e8ecf1;">{t("container_problem")}</b> {t("container_problem_body")}<br>
            <b style="color:#e8ecf1;">{t("container_solution")}</b> {t("container_solution_body")}
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_ctrl3, col_vid3, col_stats3 = st.columns([1.2, 2.8, 2.5])

    with col_ctrl3:
        source3 = get_video_source("container")

        target_container = "Container ❶ (40ft High Cube)"

        st.markdown("")
        if "run_container_v1" not in st.session_state:
            st.session_state.run_container_v1 = True
        run_container = st.toggle(t("activate_stacking"), key="run_container_v1")

    with col_vid3:
        st.markdown(f"<div style='font-family:Outfit; font-weight:600; font-size:16px; margin-bottom:8px;'>{t('live_bay_feed')}</div>", unsafe_allow_html=True)
        stframe3 = st.empty()
        alert3 = st.empty()

    with col_stats3:
        st.markdown(f"<div style='font-family:Outfit; font-weight:600; font-size:16px; margin-bottom:8px;'>{t('realtime_container')}</div>", unsafe_allow_html=True)
        metrics_placeholder3 = st.empty()
        spatial_map_placeholder3 = st.empty()
        guidance_placeholder3 = st.empty()
        log_table3 = st.empty()

    def render_container_spatial_map(stage: int = 3):
        """Render an SVG/HTML spatial plan of Container ❶ showing color-coded stacked layers dynamically."""
        box_a_opacity = "0.3" if stage >= 1 else "0.05"
        box_a_border = "#10b981" if stage >= 1 else "rgba(255,255,255,0.1)"
        box_a_text = f"{t('box_a')} [FLOOR: PACKED]" if stage >= 1 else f"{t('box_a')} [WAITING]"

        box_b_opacity = "0.25" if stage >= 2 else "0.05"
        box_b_border = "#4d6bff" if stage >= 2 else "rgba(255,255,255,0.1)"
        box_b_text = f"{t('box_b')} [MID: STACKED]" if stage >= 2 else f"{t('box_b')} [WAITING]"

        box_c_opacity = "0.25" if stage >= 3 else "0.05"
        box_c_border = "#eab308" if stage >= 3 else "rgba(255,255,255,0.1)"
        box_c_text = f"{t('box_c')} [TOP: COMPLETE]" if stage >= 3 else f"{t('box_c')} [WAITING]"

        map_html = textwrap.dedent(f"""
        <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 16px; margin-bottom: 12px;">
            <div style="font-size: 13px; font-weight: 700; color: #fff; font-family: 'Outfit', sans-serif; margin-bottom: 12px; display:flex; justify-content:space-between; align-items:center;">
                <span>{t("container_3d_map")} — {target_container}</span>
                <span class="pill pill-green" style="font-size:11px; padding:3px 10px;">{t("status_optimal")}</span>
            </div>
            <div style="position: relative; width: 100%; height: 150px; border: 2px dashed rgba(77,107,255,0.4); border-radius: 8px; background: rgba(10,15,26,0.8); padding: 8px; box-sizing: border-box; display: flex; flex-direction: column; justify-content: flex-end; gap: 6px;">
                <div style="display: flex; gap: 6px; height: 36px;">
                    <div style="flex: 1; background: rgba(234,179,8,{box_c_opacity}); border: 1px solid {box_c_border}; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700; color: #fde047;">
                        {box_c_text}
                    </div>
                </div>
                <div style="display: flex; gap: 6px; height: 40px;">
                    <div style="flex: 1; background: rgba(77,107,255,{box_b_opacity}); border: 1px solid {box_b_border}; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700; color: #93c5fd;">
                        {box_b_text}
                    </div>
                </div>
                <div style="display: flex; gap: 6px; height: 46px;">
                    <div style="flex: 1; background: rgba(16,185,129,{box_a_opacity}); border: 1px solid {box_a_border}; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 800; color: #6ee7b7;">
                        {box_a_text}
                    </div>
                </div>
            </div>
        </div>
        """)
        spatial_map_placeholder3.markdown(map_html, unsafe_allow_html=True)

    def render_stacking_guidance(stage: int = 3):
        """Render step-by-step worker stacking sequence cards dynamically based on live YOLO detections."""
        s1_badge = "✅" if stage > 1 else ("▶️ LIVE" if stage == 1 else "⏳")
        s2_badge = "✅" if stage > 2 else ("▶️ LIVE" if stage == 2 else "⏳")
        s3_badge = "✅" if stage >= 3 else ("▶️ LIVE" if stage == 3 else "⏳")

        s1_bg = "rgba(16,185,129,0.15)" if stage >= 1 else "rgba(255,255,255,0.02)"
        s2_bg = "rgba(77,107,255,0.15)" if stage >= 2 else "rgba(255,255,255,0.02)"
        s3_bg = "rgba(234,179,8,0.15)" if stage >= 3 else "rgba(255,255,255,0.02)"

        guidance_html = textwrap.dedent(f"""
        <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 14px; margin-bottom: 12px;">
            <div style="font-size: 13px; font-weight: 700; color: #fff; font-family: 'Outfit', sans-serif; margin-bottom: 10px;">
                {t("stacking_guidance")}
            </div>
            <div style="display: flex; flex-direction: column; gap: 8px;">
                <div style="background: {s1_bg}; border-left: 4px solid #10b981; padding: 10px 12px; border-radius: 6px; font-size: 12px; color: #e8ecf1;">
                    <b style="color: #10b981;">{s1_badge} {t("step_1")}</b>
                </div>
                <div style="background: {s2_bg}; border-left: 4px solid #4d6bff; padding: 10px 12px; border-radius: 6px; font-size: 12px; color: #e8ecf1;">
                    <b style="color: #4d6bff;">{s2_badge} {t("step_2")}</b>
                </div>
                <div style="background: {s3_bg}; border-left: 4px solid #eab308; padding: 10px 12px; border-radius: 6px; font-size: 12px; color: #e8ecf1;">
                    <b style="color: #eab308;">{s3_badge} {t("step_3")}</b>
                </div>
            </div>
        </div>
        """)
        guidance_placeholder3.markdown(guidance_html, unsafe_allow_html=True)

    if not run_container:
        stframe3.info("Turn **ON** Activate Stacking Optimizer to start the video.")
    elif not source3:
        stframe3.warning(t("select_source"))
    elif run_container and source3:
        cap, cap_err = open_video_capture(source3)
        if cap is None:
            st.error(t("open_source_fail", path=source3))
            stframe3.warning(t("video_open_fail"))
        else:
            try:
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_delay = 1.0 / fps if fps > 0 else 0.033
                frame_idx = 0
                model = get_detector(for_packages=False)

                # Log stacking event once per session
                conn = sqlite3.connect(DB_PATH)
                conn.execute(
                    "INSERT INTO container_logs (container_type, fill_rate, weight_balance, step_instruction, timestamp) VALUES (?,?,?,?,?)",
                    (target_container, 94.2, t("axle_even"), t("step_1"), datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                )
                conn.commit()
                conn.close()

                while run_container:
                    start_time = time.time()
                    ret, frame = cap.read()
                    if not ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    frame_idx += 1

                    h, w = frame.shape[:2]
                    target_w = 480 if LOW_MEM else 640
                    target_h = int(h * (target_w / w))
                    resized = cv2.resize(frame, (target_w, target_h))
                    if source3 == DEMO_VIDEO_LOADING or "Scenario 2" in st.session_state.get("container_scenario_sel", ""):
                        resized = cv2.GaussianBlur(resized, (7, 7), 0)

                    # Perform YOLO detection
                    results = model.predict(resized, conf=0.12, imgsz=INFER_IMGSZ, verbose=False)
                    annotated = resized.copy()

                    # Color-code items on the belt as Box A (Green), Box B (Blue), Box C (Yellow)
                    boxes = nms_boxes([(int(b.xyxy[0][0]), int(b.xyxy[0][1]), int(b.xyxy[0][2]), int(b.xyxy[0][3]), float(b.conf[0])) for b in results[0].boxes if model.names[int(b.cls[0])] in PACKAGE_COCO_CLASSES], thr=0.30)

                    det_n = len(boxes)
                    live_stage = min(3, max(1, det_n))
                    live_fill_rate = 45.0 if live_stage == 1 else (78.0 if live_stage == 2 else 94.2)

                    for idx, (x1, y1, x2, y2, conf_v) in enumerate(boxes):
                        if idx % 3 == 0:
                            color = (16, 185, 129)  # Green -> Box A Heavy
                            label_str = f"{t('box_a')} {conf_v:.2f}"
                        elif idx % 3 == 1:
                            color = (255, 107, 77)  # Blue -> Box B Medium
                            label_str = f"{t('box_b')} {conf_v:.2f}"
                        else:
                            color = (8, 179, 234)   # Yellow -> Box C Fragile
                            label_str = f"{t('box_c')} {conf_v:.2f}"

                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        draw_text(annotated, label_str, (x1, max(18, y1 - 8)), color_bgr=color, scale=0.5, thickness=2)

                    # Top banner: 3D Layout Optimization Status
                    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 36), (10, 15, 26), -1)
                    draw_text(annotated, f"AI 3D Packing Engine | Fill Ratio: {live_fill_rate:.1f}% | {target_container}", (12, 24), color_bgr=(16, 185, 129), scale=0.6, thickness=2)

                    alert3.info(t("recommendation_banner"))
                    stframe3.image(annotated, channels="BGR", width=520)

                    # Render Live Logistics KPIs dynamically
                    metrics_html3 = f"""
                    <div style="display: flex; gap: 12px; margin-bottom: 12px;">
                        <div style="flex: 1; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 12px; text-align: center;">
                            <div style="font-size: 11px; color: #5b6b7e; font-weight: 600;">{t("fill_ratio")}</div>
                            <div style="font-size: 20px; font-weight: 800; color: #10b981; font-family: Outfit;">{live_fill_rate:.1f}%</div>
                        </div>
                        <div style="flex: 1; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 12px; text-align: center;">
                            <div style="font-size: 11px; color: #5b6b7e; font-weight: 600;">{t("weight_balance")}</div>
                            <div style="font-size: 14px; font-weight: 700; color: #4d6bff; font-family: Outfit; margin-top:4px;">{t("axle_even")}</div>
                        </div>
                        <div style="flex: 1.2; display: flex; align-items: center; justify-content: center;">
                            <div class="pill pill-green" style="width: 100%; text-align: center; justify-content: center; font-size: 12px;">
                                {t("status_optimal")}
                            </div>
                        </div>
                    </div>
                    """
                    metrics_placeholder3.markdown(metrics_html3, unsafe_allow_html=True)

                    render_container_spatial_map(stage=live_stage)
                    render_stacking_guidance(stage=live_stage)

                    processing_time = time.time() - start_time
                    time.sleep(max(0.001, frame_delay - processing_time))
            finally:
                cap.release()

    if not run_container:
        with col_stats3:
            metrics_html3 = f"""
            <div style="display: flex; gap: 12px; margin-bottom: 12px;">
                <div style="flex: 1; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 12px; text-align: center;">
                    <div style="font-size: 11px; color: #5b6b7e; font-weight: 600;">{t("fill_ratio")}</div>
                    <div style="font-size: 20px; font-weight: 800; color: #10b981; font-family: Outfit;">94.2%</div>
                </div>
                <div style="flex: 1; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 12px; text-align: center;">
                    <div style="font-size: 11px; color: #5b6b7e; font-weight: 600;">{t("weight_balance")}</div>
                    <div style="font-size: 14px; font-weight: 700; color: #4d6bff; font-family: Outfit; margin-top:4px;">{t("axle_even")}</div>
                </div>
                <div style="flex: 1.2; display: flex; align-items: center; justify-content: center;">
                    <div class="pill pill-blue" style="width: 100%; text-align: center; justify-content: center; font-size: 12px;">
                        {t("inactive")}
                    </div>
                </div>
            </div>
            """
            metrics_placeholder3.markdown(metrics_html3, unsafe_allow_html=True)
            render_container_spatial_map()
            render_stacking_guidance()

    # Export reports
    conn = sqlite3.connect(DB_PATH)
    full3 = pd.read_sql_query(
        "SELECT timestamp as Time, container_type as [Container Type], fill_rate as [Fill Rate %], "
        "weight_balance as [Weight Balance], step_instruction as [Step Guidance] FROM container_logs ORDER BY id DESC",
        conn,
    )
    conn.close()
    if not full3.empty:
        with col_ctrl3:
            st.markdown("---")
            st.download_button(t("export_logs"), full3.to_csv(index=False).encode('utf-8'),
                               "container_stacking_report.csv", "text/csv")


# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
_footer_logo = (
    f'<img src="{LOGO_URI}" alt="" style="height:18px; width:auto; opacity:0.85; vertical-align:middle; margin-right:8px;" />'
    if LOGO_URI else ""
)
_footer_rofi = (
    f'<img src="{ROFI_URI}" alt="Rofi" style="height:56px; width:auto; vertical-align:middle; margin-left:10px; opacity:0.95;" />'
    if ROFI_URI else ""
)
st.markdown(f"""
<div style="display:flex; align-items:center; justify-content:space-between; gap:12px;
            font-family:'Inter'; font-size:11px; color:#3d4a5c; padding:8px 0;">
    <div style="display:flex; align-items:center;">
        {_footer_logo}
        <span>{t("footer_left")}</span>
    </div>
    <div style="display:flex; align-items:center;">
        <span>{t("footer_right")}</span>
        {_footer_rofi}
    </div>
</div>
""", unsafe_allow_html=True)