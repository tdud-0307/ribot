import math
import threading

import cv2
import mediapipe as mp
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

st.set_page_config(page_title="AI 맞춤형 보조 로봇팔", page_icon="🦾", layout="wide")
st.title("🦾 AI 기반 인체 맞춤형 웨어러블 로봇")
st.caption("첫 화면에서 웨어러블 로봇 유형을 선택하고, 웹캠으로 팔꿈치와 손가락 움직임을 추적해 3D 가상 로봇에 반영합니다.")

# 첫 화면에서 설계 유형을 선택하도록 구성
st.subheader("먼저 만들 웨어러블 로봇 유형을 선택하세요")
robot_type = st.radio(
    "로봇 유형",
    options=["soft", "extra"],
    format_func=lambda value: (
        "🧤 소프트 유연형 웨어러블 로봇팔" if value == "soft"
        else "🦾 웨어러블 엑스트라 암(추가 로봇팔)"
    ),
    horizontal=True,
    key="robot_type_choice",
    label_visibility="collapsed",
)

if robot_type == "soft":
    st.success(
        "선택한 유형: 소프트 유연형 웨어러블 로봇팔 — 착용자의 팔을 감싸거나 따라가며 "
        "움직임을 보조하는 유연한 구조를 가정합니다. 시각화에서는 팔을 감싸는 소프트 모듈을 강조합니다."
    )
else:
    st.success(
        "선택한 유형: 웨어러블 엑스트라 암 — 몸통이나 어깨 주변에 추가로 장착되는 보조 로봇팔을 가정합니다. "
        "시각화에서는 장착 지지대와 추가 팔 구조를 표시합니다. 현재 시제품에서는 사용자의 팔 동작을 입력 신호로 사용합니다."
    )

st.info(
    "이 앱은 웹캠 영상으로 관절 위치를 추정하는 시연용 프로그램입니다. "
    "픽셀 기반 길이와 2D 영상의 관절 각도는 실제 신체 치수·3D 움직임과 차이가 날 수 있습니다. "
    "현재 결과는 가상 시뮬레이션에만 적용되며 실제 보조기기 제어용으로 사용하면 안 됩니다."
)

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.measurements = {}
        self.status = "카메라를 시작하면 인식 결과가 여기에 표시됩니다."

shared_state = SharedState()


def angle_2d(a, b, c):
    """Return angle ABC in degrees, using normalized image coordinates."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    denom = math.hypot(*ba) * math.hypot(*bc)
    if denom < 1e-8:
        return None
    cosine = max(-1.0, min(1.0, (ba[0] * bc[0] + ba[1] * bc[1]) / denom))
    return math.degrees(math.acos(cosine))


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


class BodyHandProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False, model_complexity=1,
            enable_segmentation=False, min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hands = mp_hands.Hands(
            static_image_mode=False, max_num_hands=2, model_complexity=1,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        height, width = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pose_result = self.pose.process(rgb)
        hands_result = self.hands.process(rgb)
        values = {}
        status_parts = []

        if pose_result.pose_landmarks:
            lm = pose_result.pose_landmarks.landmark
            mp_drawing.draw_landmarks(image, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            def pose_point(i):
                return (lm[i].x, lm[i].y)

            def pixel_distance(i, j):
                return math.hypot((lm[i].x - lm[j].x) * width, (lm[i].y - lm[j].y) * height)

            # Shoulder=11/12, elbow=13/14, wrist=15/16
            upper_lengths, forearm_lengths, elbow_flexions = [], [], []
            for shoulder_i, elbow_i, wrist_i in [(11, 13, 15), (12, 14, 16)]:
                visibility = min(lm[shoulder_i].visibility, lm[elbow_i].visibility, lm[wrist_i].visibility)
                if visibility >= 0.35:
                    upper_lengths.append(pixel_distance(shoulder_i, elbow_i))
                    forearm_lengths.append(pixel_distance(elbow_i, wrist_i))
                    inner_angle = angle_2d(pose_point(shoulder_i), pose_point(elbow_i), pose_point(wrist_i))
                    if inner_angle is not None:
                        # A straight arm is about 180 degrees; flexion is about 0 degrees.
                        elbow_flexions.append(clamp((180.0 - inner_angle) / 150.0, 0.0, 1.0) * 150.0)

            if upper_lengths and forearm_lengths:
                values["upper_arm_px"] = sum(upper_lengths) / len(upper_lengths)
                values["forearm_px"] = sum(forearm_lengths) / len(forearm_lengths)
                if elbow_flexions:
                    values["elbow_flexion_deg"] = sum(elbow_flexions) / len(elbow_flexions)
                status_parts.append("팔·팔꿈치 인식 성공")
            else:
                status_parts.append("팔 관절이 가려졌습니다. 어깨·팔꿈치·손목을 화면에 보여 주세요.")
        else:
            status_parts.append("팔을 인식하지 못했습니다. 상체가 화면에 보이도록 해 주세요.")

        if hands_result.multi_hand_landmarks:
            all_curls = {"thumb": [], "index": [], "middle": [], "ring": [], "pinky": []}
            index_lengths, palm_lengths = [], []
            for hand_landmarks in hands_result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                h = hand_landmarks.landmark

                def hand_px(i, j):
                    return math.hypot((h[i].x - h[j].x) * width, (h[i].y - h[j].y) * height)

                index_lengths.append(hand_px(5, 8))
                palm_lengths.append(hand_px(5, 17))
                wrist = (h[0].x, h[0].y)
                # Tip-to-wrist distance relative to the finger base is a simple 2D curl estimate.
                # The thumb uses its own base/tip landmarks; the other four fingers use MCP/tip.
                finger_defs = {
                    "thumb": (2, 4, 1.25, 0.55),
                    "index": (5, 8, 1.45, 0.65),
                    "middle": (9, 12, 1.45, 0.65),
                    "ring": (13, 16, 1.40, 0.60),
                    "pinky": (17, 20, 1.35, 0.55),
                }
                for name, (base_i, tip_i, extended_ratio, ratio_range) in finger_defs.items():
                    base_d = math.hypot(h[base_i].x - wrist[0], h[base_i].y - wrist[1])
                    tip_d = math.hypot(h[tip_i].x - wrist[0], h[tip_i].y - wrist[1])
                    ratio = tip_d / max(base_d, 1e-6)
                    curl = clamp((extended_ratio - ratio) / ratio_range)
                    all_curls[name].append(curl)

            for name, curl_values in all_curls.items():
                if curl_values:
                    values[f"{name}_curl"] = sum(curl_values) / len(curl_values)
            values["finger_curl"] = sum(values.get(f"{n}_curl", 0.0) for n in all_curls) / len(all_curls)
            values["index_finger_px"] = sum(index_lengths) / len(index_lengths)
            values["palm_width_px"] = sum(palm_lengths) / len(palm_lengths)
            status_parts.append("손가락 인식 성공")
        else:
            status_parts.append("손이 보이지 않거나 인식되지 않았습니다.")

        with shared_state.lock:
            shared_state.frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            shared_state.measurements = values
            shared_state.status = " / ".join(status_parts)
        return frame.from_ndarray(image, format="bgr24")


with st.sidebar:
    st.header("⚙️ 측정 및 로봇 설정")
    st.caption("현재 선택: " + ("소프트 유연형" if robot_type == "soft" else "웨어러블 엑스트라 암"))
    cm_per_pixel = st.number_input(
        "보정 계수 (cm / pixel)", min_value=0.001, max_value=2.0,
        value=0.10, step=0.005, format="%.3f",
        help="기준 길이와 카메라 보정으로 얻은 값입니다. 기본값은 시연용 예시입니다.",
    )
    robot_scale = st.slider("로봇 크기 배율", 0.5, 1.8, 1.0, 0.05)
    motion_gain = st.slider("동작 반응 배율", 0.5, 2.0, 1.0, 0.1)
    st.caption("배율은 가상 로봇 시뮬레이션에만 적용됩니다.")

left, right = st.columns([1.05, 1])
with left:
    st.subheader("1. 실시간 웹캠 인식")
    webrtc_streamer(
        key="body-hand-tracker", mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_processor_factory=BodyHandProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    st.caption("브라우저에서 카메라 권한을 허용하세요. 웹캠 연결은 HTTPS가 필요할 수 있습니다.")
    st_autorefresh(interval=500, key="refresh_measurements")
    with shared_state.lock:
        latest_frame = shared_state.frame
        measurements = dict(shared_state.measurements)
        status = shared_state.status
    if latest_frame is not None:
        st.image(latest_frame, caption="인식된 팔·손 관절", use_container_width=True)
    st.write(f"**인식 상태:** {status}")

with right:
    st.subheader("2. 추정 치수와 동작")
    upper_px = measurements.get("upper_arm_px")
    forearm_px = measurements.get("forearm_px")
    index_px = measurements.get("index_finger_px")
    palm_px = measurements.get("palm_width_px")
    elbow_flexion = measurements.get("elbow_flexion_deg", 15.0)
    finger_curl = measurements.get("finger_curl", 0.0)

    c1, c2 = st.columns(2)
    c1.metric("상완 길이 추정", f"{upper_px * cm_per_pixel:.1f} cm" if upper_px else "—")
    c2.metric("하완 길이 추정", f"{forearm_px * cm_per_pixel:.1f} cm" if forearm_px else "—")
    c3, c4 = st.columns(2)
    c3.metric("팔꿈치 굽힘 추정", f"{elbow_flexion:.0f}°" if measurements.get("elbow_flexion_deg") is not None else "—")
    c4.metric("손가락 굽힘 추정", f"{finger_curl * 100:.0f}%" if measurements.get("finger_curl") is not None else "—")
    if index_px and palm_px:
        st.caption(f"검지 길이 추정: {index_px * cm_per_pixel:.1f} cm · 손바닥 너비 추정: {palm_px * cm_per_pixel:.1f} cm")

    st.subheader(
        "3. 소프트 유연형 웨어러블 로봇팔 시뮬레이션"
        if robot_type == "soft"
        else "3. 웨어러블 엑스트라 암 시뮬레이션"
    )
    upper_cm = max(5.0, min((upper_px * cm_per_pixel if upper_px else 30.0) * robot_scale, 60.0))
    forearm_cm = max(5.0, min((forearm_px * cm_per_pixel if forearm_px else 25.0) * robot_scale, 60.0))
    flex_rad = math.radians(max(0.0, min(elbow_flexion * motion_gain, 150.0)))
    shoulder = (0.0, 0.0, 0.0)
    elbow = (0.0, -upper_cm, 0.0)
    wrist = (forearm_cm * math.sin(flex_rad), -upper_cm - forearm_cm * math.cos(flex_rad), 0.0)

    fig = go.Figure()
    # 실제 팔의 위치를 로봇팔과 함께 그려 길이와 부착 관계를 한눈에 보여 줌.
    # 사람 팔은 로봇팔과 겹치지 않도록 Z축 방향으로 살짝 분리한 개념도임.
    human_z = -2.5 if robot_type == "soft" else -1.5
    fig.add_trace(go.Scatter3d(
        x=[shoulder[0], elbow[0], wrist[0]],
        y=[shoulder[1], elbow[1], wrist[1]],
        z=[human_z, human_z, human_z],
        mode="lines+markers",
        line=dict(color="#D99A72", width=13),
        marker=dict(color="#D99A72", size=7),
        name="사용자 팔(비교용)",
    ))
    fig.add_trace(go.Scatter3d(
        x=[shoulder[0], elbow[0], wrist[0]],
        y=[shoulder[1], elbow[1], wrist[1]],
        z=[human_z, human_z, human_z],
        mode="markers+text",
        marker=dict(size=1, opacity=0),
        text=["어깨", "팔꿈치", "손목"],
        textposition="middle right",
        showlegend=False,
        hoverinfo="skip",
    ))

    # 유형에 따라 3D 가상 구조를 다르게 표현
    if robot_type == "soft":
        # 팔을 감싸는 유연한 착용 모듈을 두꺼운 링크와 중간 밴드로 표현
        upper_width, forearm_width = 23, 20
        fig.add_trace(go.Scatter3d(
            x=[shoulder[0], elbow[0]], y=[shoulder[1], elbow[1]], z=[0, 0],
            mode="lines+markers", line=dict(width=upper_width), marker=dict(size=9),
            name="소프트 상완 모듈"
        ))
        fig.add_trace(go.Scatter3d(
            x=[elbow[0], wrist[0]], y=[elbow[1], wrist[1]], z=[0, 0],
            mode="lines+markers", line=dict(width=forearm_width), marker=dict(size=8),
            name="소프트 전완 모듈"
        ))
        # 착용 밴드 위치를 표시해 팔을 감싸는 구조임을 시각적으로 나타냄
        upper_mid = ((shoulder[0] + elbow[0]) / 2, (shoulder[1] + elbow[1]) / 2)
        forearm_mid = ((elbow[0] + wrist[0]) / 2, (elbow[1] + wrist[1]) / 2)
        fig.add_trace(go.Scatter3d(
            x=[upper_mid[0], forearm_mid[0]], y=[upper_mid[1], forearm_mid[1]], z=[0.8, 0.8],
            mode="markers", marker=dict(size=10, symbol="diamond"), name="유연 착용 밴드"
        ))
        # 밴드가 사람 팔과 로봇 모듈을 연결하는 모습을 표시
        for attach_x, attach_y in [upper_mid, forearm_mid]:
            fig.add_trace(go.Scatter3d(
                x=[attach_x, attach_x], y=[attach_y, attach_y], z=[human_z, 0.8],
                mode="lines", line=dict(color="#7C8798", width=5, dash="dash"),
                showlegend=False, hoverinfo="skip"
            ))
    else:
        # 몸통과 장착 지지대를 그려 추가 로봇팔 형태를 표현
        fig.add_trace(go.Scatter3d(
            x=[-7, -7], y=[8, -10], z=[-1, -1], mode="lines",
            line=dict(width=18), name="몸통(개념도)"
        ))
        fig.add_trace(go.Scatter3d(
            x=[-7, shoulder[0]], y=[0, shoulder[1]], z=[-1, 0], mode="lines+markers",
            line=dict(width=10), marker=dict(size=5), name="착용 장착 지지대"
        ))
        fig.add_trace(go.Scatter3d(
            x=[shoulder[0], elbow[0]], y=[shoulder[1], elbow[1]], z=[0, 0],
            mode="lines+markers", line=dict(color="#3977C3", width=14), marker=dict(size=8), name="추가 로봇 상완"
        ))
        fig.add_trace(go.Scatter3d(
            x=[elbow[0], wrist[0]], y=[elbow[1], wrist[1]], z=[0, 0],
            mode="lines+markers", line=dict(color="#3977C3", width=12), marker=dict(size=7), name="추가 로봇 전완"
        ))
        # 사람 팔과 추가 팔의 장착 지지대를 연결해 착용 관계를 명확히 표시
        for attach_x, attach_y in [shoulder, elbow]:
            fig.add_trace(go.Scatter3d(
                x=[attach_x, attach_x], y=[attach_y, attach_y], z=[human_z, 0],
                mode="lines", line=dict(color="#7C8798", width=5, dash="dash"),
                showlegend=False, hoverinfo="skip"
            ))
    fig.add_trace(go.Scatter3d(x=[elbow[0]], y=[elbow[1]], z=[0], mode="markers", marker=dict(size=10), name="로봇 팔꿈치 관절"))
    # 길이 숫자뿐 아니라 실제 모델 위에도 치수 라벨을 붙임
    upper_mid_x = (shoulder[0] + elbow[0]) / 2
    upper_mid_y = (shoulder[1] + elbow[1]) / 2
    fore_mid_x = (elbow[0] + wrist[0]) / 2
    fore_mid_y = (elbow[1] + wrist[1]) / 2
    fig.add_trace(go.Scatter3d(
        x=[upper_mid_x, fore_mid_x], y=[upper_mid_y, fore_mid_y], z=[2.2, 2.2],
        mode="text", text=[f"상완 {upper_cm:.1f} cm", f"하완 {forearm_cm:.1f} cm"],
        textfont=dict(size=12, color="#172033"), name="추정 길이", showlegend=False,
        hovertemplate="%{text}<extra></extra>"
    ))

    # Palm follows the forearm direction. Each finger bends according to its estimated curl.
    d = (math.sin(flex_rad), -math.cos(flex_rad))
    p = (math.cos(flex_rad), math.sin(flex_rad))
    palm_len = max(4.0, min(forearm_cm * 0.22, 9.0))
    palm_end = (wrist[0] + d[0] * palm_len, wrist[1] + d[1] * palm_len)
    fig.add_trace(go.Scatter3d(x=[wrist[0], palm_end[0]], y=[wrist[1], palm_end[1]], z=[0, 0], mode="lines+markers", line=dict(width=10), marker=dict(size=5), name="손바닥"))

    finger_names = ["thumb", "index", "middle", "ring", "pinky"]
    finger_labels = ["엄지", "검지", "중지", "약지", "새끼"]
    finger_lengths = [palm_len * 0.72, palm_len * 0.95, palm_len * 1.05, palm_len * 0.92, palm_len * 0.68]
    finger_offsets = [-palm_len * 0.42, -palm_len * 0.22, 0.0, palm_len * 0.22, palm_len * 0.42]
    for name, label, finger_len, offset in zip(finger_names, finger_labels, finger_lengths, finger_offsets):
        curl = measurements.get(f"{name}_curl", finger_curl)
        base = (palm_end[0] + p[0] * offset, palm_end[1] + p[1] * offset)
        # Finger segments bend backward progressively as curl increases.
        lengths = [finger_len * 0.42, finger_len * 0.33, finger_len * 0.25]
        point_x, point_y = base
        xs, ys = [point_x], [point_y]
        for segment_i, seg_len in enumerate(lengths):
            bend_deg = curl * (35 + segment_i * 55)
            bend_rad = math.radians(bend_deg)
            seg_dx = d[0] * math.cos(bend_rad) - p[0] * math.sin(bend_rad)
            seg_dy = d[1] * math.cos(bend_rad) - p[1] * math.sin(bend_rad)
            point_x += seg_dx * seg_len
            point_y += seg_dy * seg_len
            xs.append(point_x)
            ys.append(point_y)
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=[0] * len(xs), mode="lines+markers", line=dict(width=6), marker=dict(size=3), name=label))

    fig.update_layout(
        height=520, margin=dict(l=0, r=0, t=10, b=0),
        scene=dict(xaxis_title="X (cm)", yaxis_title="Y (cm)", zaxis_title="Z (cm)", aspectmode="data", camera=dict(eye=dict(x=1.6, y=1.4, z=1.25))),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True)
    if robot_type == "soft":
        st.caption("소프트 유연형 모드: 팔을 감싸는 유연 모듈을 개념적으로 표현했습니다. 실제 소재의 변형이나 공압·케이블 구동은 시뮬레이션하지 않습니다.")
    else:
        st.caption("엑스트라 암 모드: 몸통 장착 지지대와 추가 로봇팔을 개념적으로 표현했습니다. 사용자의 팔 동작은 가상 추가 팔의 입력으로 사용되며 실제 독립 제어는 구현되지 않았습니다.")
    st.caption("주황색은 사용자 팔, 파란색/굵은 모듈은 가상 로봇팔입니다. 점선은 착용 지지·밴드를 개념적으로 표시합니다. 모델 위에 상완·하완 추정 길이도 표시됩니다. 영상 기반 추정이므로 실제 관절 각도와 다를 수 있습니다.")

st.divider()
st.subheader("4. 구현된 기능과 한계")
st.markdown(
    f"- **선택한 유형:** {'소프트 유연형 웨어러블 로봇팔' if robot_type == 'soft' else '웨어러블 엑스트라 암'}.\n"
    "- **팔꿈치 동기화:** 어깨–팔꿈치–손목의 영상상 각도로 팔꿈치 굽힘 정도를 추정합니다.\n"
    "- **손가락 동기화:** 손목과 손가락 기준점·끝점의 상대 거리로 각 손가락의 굽힘 정도를 추정합니다.\n"
    "- **3D 시각화:** 추정된 관절 움직임을 Plotly 가상 로봇팔과 다섯 손가락에 반영합니다.\n"
    "- **한계:** 2D 웹캠 영상만 사용하므로 손가락이 겹치거나 옆으로 돌아가면 추정 오차가 커질 수 있습니다. 실제 로봇 모터 제어는 포함되어 있지 않습니다."
)
