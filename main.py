import math
import threading

import numpy as np

import cv2
import mediapipe as mp
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

st.set_page_config(page_title="AI 맞춤형 보조 로봇팔", page_icon="🦾", layout="wide")
st.title("🦾 AI 기반 인체 맞춤형 웨어러블 로봇")
st.caption("첫 화면에서 웨어러블 로봇 유형을 선택하고, 웹캠으로 팔꿈치와 손가락 움직임을 추적하고, 가림 상황에서는 제한적으로 추정해 3D 가상 로봇에 반영합니다.")

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
        # 마지막으로 신뢰도 높게 측정된 길이. 가려짐이 생기면 제한적으로 참고한다.
        self.last_good_arm = None
        self.missed_arm_frames = 0

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
            side_visibilities = []
            for shoulder_i, elbow_i, wrist_i in [(11, 13, 15), (12, 14, 16)]:
                sv, ev, wv = lm[shoulder_i].visibility, lm[elbow_i].visibility, lm[wrist_i].visibility
                side_visibilities.append((sv, ev, wv, shoulder_i, elbow_i, wrist_i))
                if min(sv, ev, wv) >= 0.35:
                    upper_lengths.append(pixel_distance(shoulder_i, elbow_i))
                    forearm_lengths.append(pixel_distance(elbow_i, wrist_i))
                    inner_angle = angle_2d(pose_point(shoulder_i), pose_point(elbow_i), pose_point(wrist_i))
                    if inner_angle is not None:
                        elbow_flexions.append(clamp((180.0 - inner_angle) / 150.0, 0.0, 1.0) * 150.0)

            if upper_lengths and forearm_lengths:
                values["upper_arm_px"] = float(np.mean(upper_lengths))
                values["forearm_px"] = float(np.mean(forearm_lengths))
                if elbow_flexions:
                    values["elbow_flexion_deg"] = float(np.mean(elbow_flexions))
                values["arm_estimation_mode"] = "direct"
                values["arm_error_pct"] = 8.0 if len(upper_lengths) == 2 else 12.0
                self.last_good_arm = (values["upper_arm_px"], values["forearm_px"], values.get("elbow_flexion_deg", 0.0))
                self.missed_arm_frames = 0
                status_parts.append("팔 관절 직접 인식")
            else:
                # 팔꿈치만 가려지고 어깨와 손목은 보이는 경우: 전체 길이를 비례 분할해 참고값 생성.
                inferred = []
                for sv, ev, wv, si, ei, wi in side_visibilities:
                    if sv >= 0.45 and wv >= 0.45 and ev < 0.35:
                        span = pixel_distance(si, wi)
                        if span > 5:
                            inferred.append((span * 0.46, span * 0.54))
                if inferred:
                    values["upper_arm_px"] = float(np.mean([v[0] for v in inferred]))
                    values["forearm_px"] = float(np.mean([v[1] for v in inferred]))
                    values["elbow_flexion_deg"] = 0.0
                    values["arm_estimation_mode"] = "inferred_elbow"
                    values["arm_error_pct"] = 25.0
                    status_parts.append("팔꿈치 가림: 어깨-손목 길이를 비율로 추정(오차 큼)")
                elif self.last_good_arm is not None and self.missed_arm_frames < 12:
                    self.missed_arm_frames += 1
                    values["upper_arm_px"], values["forearm_px"], values["elbow_flexion_deg"] = self.last_good_arm
                    values["arm_estimation_mode"] = "held_previous"
                    values["arm_error_pct"] = min(35.0, 15.0 + self.missed_arm_frames * 1.5)
                    status_parts.append("관절 가림: 직전 측정값 임시 유지(현재 자세와 다를 수 있음)")
                else:
                    values["arm_estimation_mode"] = "unavailable"
                    values["arm_error_pct"] = None
                    status_parts.append("팔 관절을 충분히 볼 수 없습니다. 카메라 방향을 조정해 주세요.")

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
    # 팔 관절이 현재 프레임에서 충분히 감지되지 않으면 사용자가 자세를 바로잡도록 안내
    arm_visible = (
        measurements.get("upper_arm_px") is not None
        and measurements.get("forearm_px") is not None
        and measurements.get("arm_estimation_mode") != "unavailable"
    )
    if not arm_visible:
        st.warning("팔을 보여주세요. 어깨부터 손목까지 팔이 화면 안에 들어오도록 카메라 앞에 위치해 주세요.")

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
    error_pct = measurements.get("arm_error_pct")
    if upper_px and forearm_px and error_pct is not None:
        total_cm = (upper_px + forearm_px) * cm_per_pixel
        margin_cm = total_cm * error_pct / 100.0
        mode = measurements.get("arm_estimation_mode", "direct")
        mode_label = {"direct": "관절 직접 인식", "inferred_elbow": "가려진 팔꿈치 비율 추정", "held_previous": "직전 측정값 임시 유지"}.get(mode, mode)
        st.metric("팔 길이 합계(추정)", f"{total_cm:.1f} cm", f"참고 오차 범위 ±{margin_cm:.1f} cm")
        st.caption(f"측정 방식: {mode_label} · 오차 범위는 visibility와 추정 방식에 따른 참고용 보수치이며, 통계적으로 검증된 신뢰구간은 아닙니다.")
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
# 사전 학습 모델 및 가림 상황 안내
# --------------------------------------------------
st.divider()
st.subheader("5. 사전 학습 인식 및 가림 상황 추정")
st.write(
    "사용자는 사진을 업로드하거나 학습할 필요가 없습니다. MediaPipe의 사전 학습된 Pose·Hands 모델을 "
    "앱 실행 시 사용하고, 팔꿈치가 가려졌지만 어깨와 손목이 보이는 경우에는 두 지점 사이 길이를 "
    "상완 46%, 하완 54%로 나누는 단순 비율 추정을 적용합니다."
)
st.warning(
    "가려진 팔꿈치의 실제 위치는 한 장의 2D 영상만으로 유일하게 결정할 수 없습니다. "
    "따라서 비율 추정값은 임시 시각화용이며, 실제 신체 치수나 보조기기 제어에 사용하면 안 됩니다. "
    "어깨 또는 손목까지 가려지면 추정을 중단하거나 직전 측정값을 짧게 유지합니다."
)
st.markdown(
    "**데이터 관련:** Google 이미지 검색 결과를 자동 수집해 학습 데이터로 쓰는 기능은 넣지 않았습니다. "
    "검색 이미지에는 저작권·초상권·사용 조건 문제가 있고, 대부분 팔꿈치·손목의 정답 좌표가 없어 "
    "관절 위치 학습 데이터로 바로 사용할 수 없습니다. 공개 데이터셋은 공식 배포처의 이용 조건과 "
    "관절 주석 형식을 확인한 뒤 별도 학습 파이프라인으로 준비해야 합니다."
)
st.caption(
    "현재 앱은 MediaPipe의 사전 학습 모델을 그대로 사용하며, 이번 변경은 가림 상황의 제한적 기하학적 추정과 "
    "참고 오차 범위 표시를 추가한 것입니다. MediaPipe 자체를 새로 학습한 것은 아닙니다."
)

# --------------------------------------------------
# 6. 개인 맞춤형 설계 성능 평가
# --------------------------------------------------
st.divider()
st.subheader("6. 개인 맞춤형 설계 성능 평가")
st.markdown(
    "기본형(개인 치수 미반영)과 맞춤형(사용자 팔 길이·관절 움직임 반영)의 실험 결과를 입력하면 "
    "목표 위치 도달 오차, 동작 완료 시간, 물체 잡기 성공률을 계산하고 두 설계를 비교합니다."
)
st.info(
    "아래 평가는 실험자가 측정한 값을 입력하는 방식입니다. 현재 앱은 가상 3D 시각화 프로그램이므로 "
    "실제 로봇의 이동·물체 잡기를 자동 수행하거나 센서로 측정하지 않습니다. 실제 로봇을 제작하지 않았다면 "
    "가상 시뮬레이션 결과와 실제 하드웨어 실험 결과를 구분해서 기록하세요."
)

with st.expander("평가 방법 및 측정 기준", expanded=False):
    st.markdown(
        "- **목표 위치 도달 오차(cm):** 목표 좌표와 실제 도달 좌표 사이의 직선거리. "
        "오차 = √((목표 X−실제 X)² + (목표 Y−실제 Y)²)\n"
        "- **동작 완료 시간(s):** 동작 시작부터 목표 위치에 도달하거나 물체 잡기 동작이 끝날 때까지의 시간\n"
        "- **물체 잡기 성공률(%):** 성공 횟수 ÷ 전체 시도 횟수 × 100\n"
        "- 각 조건에서 같은 목표 위치, 같은 물체, 같은 카메라·실험 환경을 유지하고 여러 번 반복하면 비교가 더 공정합니다."
    )

target_col1, target_col2 = st.columns(2)
with target_col1:
    target_x = st.number_input("공통 목표 위치 X (cm)", value=20.0, step=1.0, key="eval_target_x")
with target_col2:
    target_y = st.number_input("공통 목표 위치 Y (cm)", value=15.0, step=1.0, key="eval_target_y")

st.caption("각 조건에서 실험한 실제 도달 좌표와 시간, 물체 잡기 결과를 입력하세요.")
base_col, custom_col = st.columns(2)

def evaluation_inputs(prefix, title, default_x, default_y, default_time, default_success):
    with st.container(border=True):
        st.markdown(f"**{title}**")
        x = st.number_input("실제 도달 X (cm)", value=default_x, step=0.5, key=f"{prefix}_x")
        y = st.number_input("실제 도달 Y (cm)", value=default_y, step=0.5, key=f"{prefix}_y")
        elapsed = st.number_input("동작 완료 시간 (초)", min_value=0.0, value=default_time, step=0.1, key=f"{prefix}_time")
        successes = st.number_input("물체 잡기 성공 횟수", min_value=0, value=default_success, step=1, key=f"{prefix}_success")
        trials = st.number_input("물체 잡기 전체 시도 횟수", min_value=1, value=5, step=1, key=f"{prefix}_trials")
    return x, y, elapsed, int(successes), int(trials)

with base_col:
    base_x, base_y, base_time, base_success, base_trials = evaluation_inputs(
        "baseline", "기본형 설계 (개인 맞춤 미적용)", target_x + 3.0, target_y + 2.0, 8.0, 3
    )
with custom_col:
    custom_x, custom_y, custom_time, custom_success, custom_trials = evaluation_inputs(
        "custom", "개인 맞춤형 설계 (개인 치수 반영)", target_x + 1.0, target_y + 1.0, 6.5, 4
    )

base_error = math.hypot(target_x - base_x, target_y - base_y)
custom_error = math.hypot(target_x - custom_x, target_y - custom_y)
base_rate = base_success / base_trials * 100.0
custom_rate = custom_success / custom_trials * 100.0


def improvement_percent(before, after):
    if before <= 0:
        return None
    return (before - after) / before * 100.0

error_change = improvement_percent(base_error, custom_error)
time_change = improvement_percent(base_time, custom_time)
rate_change = custom_rate - base_rate

st.markdown("#### 평가 결과 비교")
result_rows = [
    {"평가 항목": "목표 위치 도달 오차", "기본형": f"{base_error:.2f} cm", "맞춤형": f"{custom_error:.2f} cm",
     "변화": (f"{error_change:+.1f}% 개선" if error_change is not None and error_change >= 0 else (f"{abs(error_change):.1f}% 증가" if error_change is not None else "기준 오차 0"))},
    {"평가 항목": "동작 완료 시간", "기본형": f"{base_time:.2f} 초", "맞춤형": f"{custom_time:.2f} 초",
     "변화": (f"{time_change:+.1f}% 단축" if time_change is not None and time_change >= 0 else (f"{abs(time_change):.1f}% 증가" if time_change is not None else "기준 시간 0"))},
    {"평가 항목": "물체 잡기 성공률", "기본형": f"{base_rate:.1f}% ({base_success}/{base_trials})", "맞춤형": f"{custom_rate:.1f}% ({custom_success}/{custom_trials})",
     "변화": f"{rate_change:+.1f}%p"},
]
st.table(result_rows)

m1, m2, m3 = st.columns(3)
m1.metric("맞춤형 도달 오차", f"{custom_error:.2f} cm", f"기본형 대비 {error_change:.1f}% 개선" if error_change is not None and error_change >= 0 else (f"기본형 대비 {abs(error_change):.1f}% 증가" if error_change is not None else "기본형 오차 0 cm"))
m2.metric("맞춤형 완료 시간", f"{custom_time:.2f} 초", f"기본형 대비 {time_change:.1f}% 단축" if time_change is not None and time_change >= 0 else (f"기본형 대비 {abs(time_change):.1f}% 증가" if time_change is not None else "기본형 시간 0초"))
m3.metric("맞춤형 잡기 성공률", f"{custom_rate:.1f}%", f"기본형 대비 {rate_change:+.1f}%p")

if custom_error < base_error and custom_time < base_time and custom_rate > base_rate:
    st.success("입력한 실험 결과에서는 개인 맞춤형 설계가 세 평가 항목 모두에서 더 좋은 결과를 보였습니다.")
else:
    st.warning("입력한 결과만으로는 개인 맞춤형 설계가 모든 항목에서 우수하다고 보기 어렵습니다. 반복 실험 후 평균값으로 다시 비교하세요.")

import csv
from io import StringIO
csv_buffer = StringIO()
writer = csv.DictWriter(csv_buffer, fieldnames=["평가 항목", "기본형", "맞춤형", "변화"])
writer.writeheader()
writer.writerows(result_rows)
st.download_button(
    "평가 결과 CSV 다운로드", data=csv_buffer.getvalue().encode("utf-8-sig"),
    file_name="robot_arm_performance_evaluation.csv", mime="text/csv",
)
st.caption("화면에 처음 표시되는 숫자는 입력 예시일 뿐 실제 실험 결과가 아닙니다. 본인 실험값으로 바꿔서 사용하세요.")
