import math
import threading
import zipfile
from io import BytesIO

import numpy as np

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



def add_solid_segment(fig, start, end, radius, color, name, opacity=1.0, sides=12):
    """Draw a solid 3D cylindrical/tapered robot component between two 3D points."""
    ax, ay, az = start
    bx, by, bz = end
    vx, vy, vz = bx-ax, by-ay, bz-az
    length = math.sqrt(vx*vx + vy*vy + vz*vz)
    if length < 1e-6:
        return
    ux, uy, uz = vx/length, vy/length, vz/length
    # Build two perpendicular vectors around the segment axis.
    ref = (0.0, 0.0, 1.0) if abs(uz) < 0.9 else (0.0, 1.0, 0.0)
    nx, ny, nz = uy*ref[2]-uz*ref[1], uz*ref[0]-ux*ref[2], ux*ref[1]-uy*ref[0]
    norm = math.sqrt(nx*nx + ny*ny + nz*nz) or 1.0
    nx, ny, nz = nx/norm, ny/norm, nz/norm
    mx, my, mz = uy*nz-uz*ny, uz*nx-ux*nz, ux*ny-uy*nx
    xs, ys, zs = [], [], []
    for point, r in ((start, radius*0.88), (end, radius)):
        for i in range(sides):
            a = 2*math.pi*i/sides
            xs.append(point[0] + r*(math.cos(a)*nx + math.sin(a)*mx))
            ys.append(point[1] + r*(math.cos(a)*ny + math.sin(a)*my))
            zs.append(point[2] + r*(math.cos(a)*nz + math.sin(a)*mz))
    ii, jj, kk = [], [], []
    for i in range(sides):
        j = (i+1) % sides
        ii.extend([i, i, sides+i]); jj.extend([j, sides+j, sides+j]); kk.extend([sides+j, sides+i, j])
    fig.add_trace(go.Mesh3d(x=xs, y=ys, z=zs, i=ii, j=jj, k=kk, color=color, opacity=opacity, name=name, flatshading=True, hovertemplate=name+'<extra></extra>'))
    # End caps
    for offset, point in ((0, start), (sides, end)):
        center = len(xs); xs.append(point[0]); ys.append(point[1]); zs.append(point[2])
        for i in range(sides):
            j = (i+1) % sides
            ii.append(center); jj.append(offset+i); kk.append(offset+j)
    # Update trace with capped geometry
    fig.data[-1].update(x=xs, y=ys, z=zs, i=ii, j=jj, k=kk)


def add_joint(fig, point, radius, color, name):
    fig.add_trace(go.Scatter3d(x=[point[0]], y=[point[1]], z=[point[2]], mode='markers',
        marker=dict(size=radius, color=color, line=dict(color='#333333', width=1)), name=name))


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
        # 참고 이미지처럼 팔을 감싸는 검정색 유연 슬리브와 손등/손가락 보조 구조를 3D로 표현
        add_solid_segment(fig, (shoulder[0], shoulder[1], 0.45), (elbow[0], elbow[1], 0.45), max(2.0, upper_cm*0.085), '#20252C', '소프트 상완 슬리브')
        add_solid_segment(fig, (elbow[0], elbow[1], 0.45), (wrist[0], wrist[1], 0.45), max(1.8, forearm_cm*0.075), '#303640', '소프트 전완 슬리브')
        add_solid_segment(fig, (wrist[0], wrist[1], 0.5), (wrist[0] + 4*math.sin(flex_rad), wrist[1] - 4*math.cos(flex_rad), 0.5), 1.4, '#171B20', '손등 보조 모듈')
        add_joint(fig, elbow, 7, '#171B20', '팔꿈치 힌지')
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
        # 참고 이미지처럼 상완의 주황색 외피, 기계식 팔꿈치, 검정 전완 하우징을 입체적으로 표현
        add_solid_segment(fig, (shoulder[0], shoulder[1], 0.45), (elbow[0], elbow[1], 0.45), max(2.4, upper_cm*0.095), '#B9572B', '엑스트라 암 상완 외피')
        add_solid_segment(fig, (elbow[0], elbow[1], 0.45), (wrist[0], wrist[1], 0.45), max(2.0, forearm_cm*0.085), '#20242A', '엑스트라 암 전완 하우징')
        add_joint(fig, elbow, 9, '#AEB5BD', '기계식 팔꿈치 관절')
        add_joint(fig, wrist, 7, '#BFC7CE', '손목 연결 관절')
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
        for attach_point in [shoulder, elbow]:
            # shoulder/elbow are 3D coordinates (x, y, z); use x and y for the attachment line.
            attach_x, attach_y = attach_point[0], attach_point[1]
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

    # Detailed robotic hand: palm shell, wrist cuff, knuckle blocks, three-part fingers,
    # thumb actuator, cable/tendon guides and fingertip pads. All dimensions scale with arm length.
    d = (math.sin(flex_rad), -math.cos(flex_rad))
    pvec = (math.cos(flex_rad), math.sin(flex_rad))
    palm_len = max(4.0, min(forearm_cm * 0.22, 9.0))
    palm_width = palm_len * 0.58
    palm_start = (wrist[0] + d[0] * 0.8, wrist[1] + d[1] * 0.8)
    palm_end = (wrist[0] + d[0] * palm_len, wrist[1] + d[1] * palm_len)
    robot_dark = '#171B20' if robot_type == 'soft' else '#24282D'
    robot_mid = '#343B43' if robot_type == 'soft' else '#343A42'
    robot_metal = '#AAB4BE'
    robot_accent = '#20252B' if robot_type == 'soft' else '#B9572B'

    # Wrist cuff and a layered palm housing (the layered rails make it look less like a wireframe).
    add_solid_segment(fig, (wrist[0]-pvec[0]*palm_width*0.48, wrist[1]-pvec[1]*palm_width*0.48, 0.2),
                      (wrist[0]+pvec[0]*palm_width*0.48, wrist[1]+pvec[1]*palm_width*0.48, 0.2),
                      max(0.8, palm_width*0.13), robot_metal, '손목 금속 커프', sides=10)
    add_solid_segment(fig, (palm_start[0], palm_start[1], 0.1), (palm_end[0], palm_end[1], 0.1),
                      max(1.0, palm_width*0.26), robot_dark, '손바닥 외장 하우징', sides=12)
    # Palm side rails and central inset plate.
    for side in (-1, 1):
        sx = palm_start[0] + pvec[0] * palm_width * 0.34 * side
        sy = palm_start[1] + pvec[1] * palm_width * 0.34 * side
        ex = palm_end[0] + pvec[0] * palm_width * 0.34 * side
        ey = palm_end[1] + pvec[1] * palm_width * 0.34 * side
        add_solid_segment(fig, (sx, sy, 0.48), (ex, ey, 0.48), max(0.35, palm_width*0.055), robot_mid, '손바닥 측면 프레임', sides=8)
    add_solid_segment(fig, (palm_start[0]+d[0]*palm_len*0.22, palm_start[1]+d[1]*palm_len*0.22, 0.55),
                      (palm_end[0]-d[0]*palm_len*0.14, palm_end[1]-d[1]*palm_len*0.14, 0.55),
                      max(0.3, palm_width*0.07), robot_accent, '손바닥 중앙 커버', sides=8)

    finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
    finger_labels = ['엄지', '검지', '중지', '약지', '새끼']
    finger_lengths = [palm_len*0.68, palm_len*0.86, palm_len*0.96, palm_len*0.85, palm_len*0.66]
    finger_offsets = [-palm_width*0.48, -palm_width*0.29, -palm_width*0.08, palm_width*0.15, palm_width*0.36]
    for fi, (name, label, finger_len, offset) in enumerate(zip(finger_names, finger_labels, finger_lengths, finger_offsets)):
        curl = max(0.0, min(1.0, measurements.get(f'{name}_curl', finger_curl)))
        # Thumb branches sideways from the palm; the other fingers fan out slightly.
        base_x = palm_end[0] + pvec[0] * offset
        base_y = palm_end[1] + pvec[1] * offset
        if name == 'thumb':
            base_x -= d[0] * palm_len * 0.12
            base_y -= d[1] * palm_len * 0.12
            direction = (d[0]*0.55 - pvec[0]*0.82, d[1]*0.55 - pvec[1]*0.82)
            lengths = [finger_len*0.43, finger_len*0.34, finger_len*0.23]
        else:
            fan = (fi-2) * 0.075
            direction = (d[0] + pvec[0]*fan, d[1] + pvec[1]*fan)
            norm = math.hypot(*direction) or 1.0
            direction = (direction[0]/norm, direction[1]/norm)
            lengths = [finger_len*0.43, finger_len*0.32, finger_len*0.25]
        pts = [(base_x, base_y, 0.38)]
        px, py = base_x, base_y
        for si, seg_len in enumerate(lengths):
            # Each phalanx has its own bend; curl drives the three linked segments.
            bend = math.radians(curl * (18 + si*34))
            dx = direction[0]*math.cos(bend) - pvec[0]*math.sin(bend)
            dy = direction[1]*math.cos(bend) - pvec[1]*math.sin(bend)
            px += dx*seg_len
            py += dy*seg_len
            pts.append((px, py, 0.38 + (0.06 if si == 1 else 0.0)))
        # Distinct proximal, middle and distal phalanges with visible hinge joints.
        radii = [max(0.48, palm_width*0.105), max(0.40, palm_width*0.087), max(0.34, palm_width*0.073)]
        for si in range(3):
            color = robot_mid if si == 0 else (robot_dark if si == 1 else '#4A535D')
            add_solid_segment(fig, pts[si], pts[si+1], radii[si], color, f'{label} {si+1}節', sides=8)
            if si < 2:
                add_joint(fig, pts[si+1], 4.5, robot_metal, f'{label} 관절 {si+1}')
        add_joint(fig, pts[0], 4.0, robot_metal, f'{label} 손가락 기부')
        # Soft contact pad at each fingertip.
        add_joint(fig, pts[-1], 4.8, '#69747F', f'{label} 끝 패드')
        # Small dorsal tendon/cable line running over the finger.
        mid_pts = [pts[0], pts[1], pts[2], pts[3]]
        fig.add_trace(go.Scatter3d(x=[q[0] for q in mid_pts], y=[q[1] for q in mid_pts],
            z=[q[2]+0.34 for q in mid_pts], mode='lines',
            line=dict(color='#C5CDD4', width=2), name=f'{label} 구동 케이블',
            showlegend=(fi == 0), hoverinfo='skip'))

    # Small visible actuator pack on the back of the hand and paired mounting straps.
    add_solid_segment(fig, (palm_start[0]+pvec[0]*palm_width*0.22, palm_start[1]+pvec[1]*palm_width*0.22, 0.8),
                      (palm_start[0]+pvec[0]*palm_width*0.22+d[0]*palm_len*0.35,
                       palm_start[1]+pvec[1]*palm_width*0.22+d[1]*palm_len*0.35, 0.8),
                      max(0.42, palm_width*0.10), robot_metal, '손등 구동 링크', sides=8)
    for strap_at in (0.28, 0.72):
        sx = wrist[0] + d[0]*palm_len*strap_at
        sy = wrist[1] + d[1]*palm_len*strap_at
        add_solid_segment(fig, (sx-pvec[0]*palm_width*0.48, sy-pvec[1]*palm_width*0.48, 0.72),
                          (sx+pvec[0]*palm_width*0.48, sy+pvec[1]*palm_width*0.48, 0.72),
                          max(0.25, palm_width*0.045), robot_metal, '손등 고정 스트랩', sides=8)

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


# --------------------------------------------------
# 데이터셋 업로드 및 사용자 정의 학습 실험실
# --------------------------------------------------
st.divider()
st.subheader("5. 데이터셋 학습 실험실: 반팔·긴팔·가림 상황 분류")
st.write(
    "공개 데이터셋 또는 직접 모은 이미지를 클래스별 폴더로 묶은 ZIP 파일을 올려 "
    "영상의 시야 상태(예: 팔이 잘 보임/긴팔/팔 가림/손 가림)를 분류하는 실험용 모델을 학습합니다. "
    "이 모델은 MediaPipe 관절 검출기 자체를 재학습하는 것은 아니며, 인식 환경을 분류해 주는 보조 모델입니다."
)
st.warning(
    "이미지는 본인 또는 사용 허가를 받은 데이터만 업로드하세요. ZIP 내부 폴더 이름이 정답 라벨이 됩니다. "
    "각 클래스에 다양한 사람·배경·조명 사진을 충분히 넣고, 같은 연속 영상의 비슷한 프레임만 몰아넣지 마세요."
)

with st.expander("📁 학습 ZIP 파일 구조 예시", expanded=False):
    st.code(
        "dataset.zip\n"
        "├── 팔_잘보임/\n"
        "│   ├── image001.jpg\n"
        "│   └── image002.jpg\n"
        "├── 긴팔/\n"
        "│   ├── image001.jpg\n"
        "│   └── image002.jpg\n"
        "├── 팔_가림/\n"
        "└── 손_가림/",
        language="text",
    )
    st.caption("최소 2개 클래스가 필요합니다. 테스트용으로는 클래스마다 10장 이상을 권장하며, 더 많고 다양한 이미지일수록 좋습니다.")

training_zip = st.file_uploader("클래스별 이미지 폴더가 들어 있는 ZIP 업로드", type=["zip"], key="visibility_dataset_zip")


def image_to_hog(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    descriptor = cv2.HOGDescriptor(
        _winSize=(128, 128), _blockSize=(16, 16), _blockStride=(8, 8),
        _cellSize=(8, 8), _nbins=9
    )
    return descriptor.compute(gray).reshape(-1).astype(np.float32)


def train_visibility_model(zip_bytes):
    x_rows, y_rows, class_counts = [], [], {}
    supported = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    with zipfile.ZipFile(BytesIO(zip_bytes)) as archive:
        for member in archive.infolist():
            if member.is_dir() or not member.filename.lower().endswith(supported):
                continue
            parts = [part for part in member.filename.replace("\\", "/").split("/") if part]
            # Root folder is treated as class label; skip files without a class folder.
            if len(parts) < 2:
                continue
            label = parts[-2]
            raw = archive.read(member)
            arr = np.frombuffer(raw, dtype=np.uint8)
            image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if image_bgr is None:
                continue
            try:
                x_rows.append(image_to_hog(image_bgr))
                y_rows.append(label)
                class_counts[label] = class_counts.get(label, 0) + 1
            except Exception:
                continue
    if len(class_counts) < 2:
        raise ValueError("클래스 폴더가 2개 이상 필요합니다. ZIP 안에 라벨별 폴더를 만들어 주세요.")
    too_small = {k: v for k, v in class_counts.items() if v < 2}
    if too_small:
        raise ValueError("각 클래스에 읽을 수 있는 이미지가 최소 2장씩 필요합니다. 부족한 클래스: " + str(too_small))
    labels = sorted(class_counts.keys())
    label_to_id = {label: i for i, label in enumerate(labels)}
    y = np.array([label_to_id[label] for label in y_rows], dtype=np.int32)
    x = np.vstack(x_rows).astype(np.float32)
    # Hold out one fifth per class for a basic validation estimate when enough samples exist.
    rng = np.random.default_rng(42)
    train_indices, test_indices = [], []
    for class_id in range(len(labels)):
        class_indices = np.where(y == class_id)[0]
        rng.shuffle(class_indices)
        if len(class_indices) >= 5:
            n_test = max(1, int(round(len(class_indices) * 0.2)))
            test_indices.extend(class_indices[:n_test].tolist())
            train_indices.extend(class_indices[n_test:].tolist())
        else:
            train_indices.extend(class_indices.tolist())
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(0.5)
    svm.train(x[train_indices], cv2.ml.ROW_SAMPLE, y[train_indices])
    validation_accuracy = None
    if test_indices:
        _, predicted = svm.predict(x[test_indices])
        validation_accuracy = float(np.mean(predicted.reshape(-1).astype(np.int32) == y[test_indices]))
    # Refit using all available examples for deployment after measuring holdout accuracy.
    final_svm = cv2.ml.SVM_create()
    final_svm.setType(cv2.ml.SVM_C_SVC)
    final_svm.setKernel(cv2.ml.SVM_LINEAR)
    final_svm.setC(0.5)
    final_svm.train(x, cv2.ml.ROW_SAMPLE, y)
    return final_svm, labels, class_counts, len(x_rows), validation_accuracy

if training_zip is not None:
    if st.button("🧠 업로드한 데이터로 학습하기", type="primary", key="train_visibility_model"):
        try:
            with st.spinner("이미지 특징을 추출하고 분류 모델을 학습하는 중..."):
                model, labels, counts, n_images, validation_accuracy = train_visibility_model(training_zip.getvalue())
            st.session_state["visibility_model"] = model
            st.session_state["visibility_labels"] = labels
            st.session_state["visibility_counts"] = counts
            st.session_state["visibility_trained"] = True
            st.session_state["visibility_validation_accuracy"] = validation_accuracy
            st.success(f"학습 완료: 이미지 {n_images}장, 클래스 {len(labels)}개")
            if validation_accuracy is not None:
                st.metric("검증 이미지 분류 정확도(간이 평가)", f"{validation_accuracy * 100:.1f}%")
            else:
                st.info("클래스별 이미지가 적어 별도 검증 정확도를 계산하지 않았습니다. 클래스마다 5장 이상을 권장합니다.")
        except Exception as exc:
            st.error(f"학습하지 못했습니다: {exc}")

if st.session_state.get("visibility_trained", False):
    st.markdown("**학습 데이터 구성**")
    st.dataframe(
        [{"클래스(폴더명)": name, "이미지 수": count} for name, count in st.session_state["visibility_counts"].items()],
        use_container_width=True, hide_index=True
    )
    with shared_state.lock:
        frame_for_classification = None if shared_state.frame is None else shared_state.frame.copy()
    if frame_for_classification is not None:
        try:
            # Shared frame is RGB; HOG preprocessing expects BGR.
            bgr_for_classification = cv2.cvtColor(frame_for_classification, cv2.COLOR_RGB2BGR)
            features = image_to_hog(bgr_for_classification).reshape(1, -1)
            _, pred = st.session_state["visibility_model"].predict(features)
            pred_id = int(pred[0][0])
            labels = st.session_state["visibility_labels"]
            if 0 <= pred_id < len(labels):
                st.metric("현재 웹캠 장면 분류", labels[pred_id])
                st.caption("이 값은 업로드한 이미지와의 시각적 유사도를 이용한 실험적 분류 결과입니다. 신뢰도 점수나 관절 위치의 정확도를 보장하지 않습니다.")
        except Exception as exc:
            st.warning(f"현재 화면 분류를 수행하지 못했습니다: {exc}")
    else:
        st.info("웹캠을 켜면 학습된 분류 모델이 현재 장면을 분류합니다.")
    if st.button("학습 모델 초기화", key="reset_visibility_model"):
        for key in ("visibility_model", "visibility_labels", "visibility_counts", "visibility_trained", "visibility_validation_accuracy"):
            st.session_state.pop(key, None)
        st.rerun()

st.caption(
    "한계: 이 실험실은 HOG 이미지 특징 + 선형 SVM으로 클래스별 장면을 분류합니다. "
    "가려진 손가락의 실제 관절을 복원하거나 MediaPipe의 손·팔 랜드마크 모델을 재학습하지는 않습니다. "
    "학습 정확도를 제대로 평가하려면 별도의 테스트 이미지로 검증해야 합니다."
)
