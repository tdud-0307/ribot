import math
import threading

import cv2
import mediapipe as mp
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

st.set_page_config(
    page_title="AI 맞춤형 보조 로봇팔",
    page_icon="🦾",
    layout="wide",
)

st.title("🦾 AI 기반 인체 맞춤형 보조 로봇팔")
st.caption("웹캠으로 팔·손 관절을 인식하고, 신체 치수 추정값에 맞춰 가상 로봇팔의 크기와 동작을 조정하는 프로토타입입니다.")

st.info(
    "측정 안내: 웹캠 영상의 픽셀만으로 실제 cm를 정확히 알 수는 없습니다. "
    "아래 보정 계수는 기준 물체나 별도 보정으로 얻은 값이어야 합니다. "
    "현재 버전의 cm 값은 보정 계수를 적용한 추정치이며 의료·안전 목적의 정밀 측정값이 아닙니다."
)

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.measurements = {}
        self.status = "카메라를 시작하면 관절 인식 결과가 여기에 표시됩니다."

shared_state = SharedState()

class BodyHandProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
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
            mp_drawing.draw_landmarks(
                image,
                pose_result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
            )

            # MediaPipe pose landmark indices:
            # 11 left shoulder, 13 left elbow, 15 left wrist
            # 12 right shoulder, 14 right elbow, 16 right wrist
            def pixel_distance(i, j):
                x1, y1 = lm[i].x * width, lm[i].y * height
                x2, y2 = lm[j].x * width, lm[j].y * height
                return math.hypot(x2 - x1, y2 - y1)

            values["upper_arm_px"] = (pixel_distance(11, 13) + pixel_distance(12, 14)) / 2
            values["forearm_px"] = (pixel_distance(13, 15) + pixel_distance(14, 16)) / 2
            values["left_upper_arm_px"] = pixel_distance(11, 13)
            values["left_forearm_px"] = pixel_distance(13, 15)
            values["right_upper_arm_px"] = pixel_distance(12, 14)
            values["right_forearm_px"] = pixel_distance(14, 16)
            status_parts.append("팔 인식 성공")
        else:
            status_parts.append("팔을 인식하지 못했습니다. 어깨·팔꿈치·손목이 화면에 보이도록 해 주세요.")

        if hands_result.multi_hand_landmarks:
            finger_lengths = []
            palm_lengths = []
            for hand_landmarks in hands_result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                )
                h = hand_landmarks.landmark

                def hand_px(i, j):
                    x1, y1 = h[i].x * width, h[i].y * height
                    x2, y2 = h[j].x * width, h[j].y * height
                    return math.hypot(x2 - x1, y2 - y1)

                # Index finger MCP (5) to tip (8); palm width (5 to 17)
                finger_lengths.append(hand_px(5, 8))
                palm_lengths.append(hand_px(5, 17))
            values["index_finger_px"] = sum(finger_lengths) / len(finger_lengths)
            values["palm_width_px"] = sum(palm_lengths) / len(palm_lengths)
            status_parts.append("손 인식 성공")
        else:
            status_parts.append("손이 보이지 않거나 손을 인식하지 못했습니다.")

        with shared_state.lock:
            shared_state.frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            shared_state.measurements = values
            shared_state.status = " / ".join(status_parts)

        return frame.from_ndarray(image, format="bgr24")


with st.sidebar:
    st.header("⚙️ 측정 및 로봇 설정")
    cm_per_pixel = st.number_input(
        "보정 계수 (cm / pixel)",
        min_value=0.001,
        max_value=2.0,
        value=0.10,
        step=0.005,
        format="%.3f",
        help="기준 길이가 있는 물체와 카메라 보정으로 얻은 cm/pixel 값을 입력하세요. 기본값은 예시용이며 실제 측정값이 아닙니다.",
    )
    robot_scale = st.slider("로봇 크기 배율", 0.5, 1.8, 1.0, 0.05)
    motion_gain = st.slider("손 움직임 반응 배율", 0.5, 2.0, 1.0, 0.1)
    st.caption("이 배율은 가상 로봇 시뮬레이션에만 적용됩니다.")

left, right = st.columns([1.05, 1])

with left:
    st.subheader("1. 웹캠 신체 인식")
    webrtc_ctx = webrtc_streamer(
        key="body-hand-tracker",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        video_processor_factory=BodyHandProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    st.caption("브라우저에서 카메라 권한을 허용하세요. 배포 환경에 따라 카메라 연결에 HTTPS가 필요합니다.")

    st_autorefresh(interval=700, key="refresh_measurements")
    with shared_state.lock:
        latest_frame = shared_state.frame
        measurements = dict(shared_state.measurements)
        status = shared_state.status

    if latest_frame is not None:
        st.image(latest_frame, caption="인식된 관절", use_container_width=True)
    st.write(f"**인식 상태:** {status}")

with right:
    st.subheader("2. 추정 신체 치수")
    if measurements:
        upper = measurements.get("upper_arm_px")
        forearm = measurements.get("forearm_px")
        index = measurements.get("index_finger_px")
        palm = measurements.get("palm_width_px")

        m1, m2 = st.columns(2)
        m1.metric("상완 길이 추정", f"{upper * cm_per_pixel:.1f} cm" if upper else "—")
        m2.metric("하완 길이 추정", f"{forearm * cm_per_pixel:.1f} cm" if forearm else "—")
        m3, m4 = st.columns(2)
        m3.metric("검지 길이 추정", f"{index * cm_per_pixel:.1f} cm" if index else "—")
        m4.metric("손바닥 너비 추정", f"{palm * cm_per_pixel:.1f} cm" if palm else "—")
        st.caption("상완은 어깨–팔꿈치, 하완은 팔꿈치–손목, 검지는 검지 관절 시작점–끝점의 영상상 거리로 추정합니다.")
    else:
        st.warning("아직 관절 데이터가 없습니다. 카메라를 시작하고 상체와 손을 화면 안에 넣어 주세요.")

    st.subheader("3. 맞춤형 3D 로봇팔")
    if measurements and measurements.get("upper_arm_px") and measurements.get("forearm_px"):
        upper_cm = measurements["upper_arm_px"] * cm_per_pixel * robot_scale
        forearm_cm = measurements["forearm_px"] * cm_per_pixel * robot_scale
        # Cap values for a readable demo visualization
        upper_cm = max(5, min(upper_cm, 60))
        forearm_cm = max(5, min(forearm_cm, 60))
    else:
        upper_cm, forearm_cm = 30.0 * robot_scale, 25.0 * robot_scale

    # A simple illustrative 3D articulated arm, not a physical robot model.
    shoulder = (0, 0, 0)
    elbow = (0, -upper_cm, 0)
    wrist = (forearm_cm * 0.25 * motion_gain, -(upper_cm + forearm_cm * 0.75), 0)
    gripper_tip = (forearm_cm * 0.55 * motion_gain, -(upper_cm + forearm_cm), 0)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=[shoulder[0], elbow[0], wrist[0], gripper_tip[0]],
        y=[shoulder[1], elbow[1], wrist[1], gripper_tip[1]],
        z=[shoulder[2], elbow[2], wrist[2], gripper_tip[2]],
        mode="lines+markers",
        line=dict(width=12),
        marker=dict(size=[8, 7, 6, 5]),
        name="로봇팔",
    ))
    fig.add_trace(go.Scatter3d(
        x=[gripper_tip[0] - 2, gripper_tip[0] + 2],
        y=[gripper_tip[1] - 1, gripper_tip[1] - 1],
        z=[0, 0],
        mode="lines",
        line=dict(width=7),
        name="그리퍼 예시",
    ))
    fig.update_layout(
        height=470,
        margin=dict(l=0, r=0, t=10, b=0),
        scene=dict(
            xaxis_title="X (cm)",
            yaxis_title="Y (cm)",
            zaxis_title="Z (cm)",
            aspectmode="data",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"**가상 상완 길이:** {upper_cm:.1f} cm · **가상 하완 길이:** {forearm_cm:.1f} cm")
    st.caption("현재 3D 모델은 링크 길이가 변하는 개념 시뮬레이션입니다. 실제 역기구학이나 실제 로봇 하드웨어 제어는 아직 포함하지 않았습니다.")

st.divider()
st.subheader("4. 다음 개발 목표")
st.markdown(
    """
    - 기준 물체를 자동으로 인식해 cm/pixel 보정 계수를 계산하기
    - 손가락 굽힘 각도를 이용해 가상 그리퍼를 열고 닫기
    - 실제 3D 관절 모델에 역기구학을 적용하기
    - 기본형과 맞춤형 로봇의 목표 위치 오차를 비교하기
    """
)
