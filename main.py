import math
import os
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import cv2
import mediapipe as mp


def prepare_writable_pose_model():
    """
    MediaPipe 패키지 자체는 site-packages에서 그대로 불러오고,
    쓰기 권한이 필요한 Pose 모델 파일만 /tmp에 준비한다.
    패키지 전체를 복사하지 않아 네이티브 바인딩 구조가 깨지는 문제를 방지한다.
    """
    model_dir = Path(tempfile.gettempdir()) / "ribot_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "pose_landmark_lite.tflite"

    if not model_path.is_file() or model_path.stat().st_size < 100_000:
        temp_path = model_path.with_suffix(".tflite.part")
        model_url = "https://storage.googleapis.com/mediapipe-assets/pose_landmark_lite.tflite"
        try:
            request = urllib.request.Request(
                model_url,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(request, timeout=60) as response, open(temp_path, "wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)

            if temp_path.stat().st_size < 100_000:
                raise RuntimeError("다운로드된 Pose 모델 파일의 크기가 너무 작습니다.")
            os.replace(temp_path, model_path)
        except Exception as exc:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise RuntimeError(
                "Pose 모델을 준비하지 못했습니다. Streamlit Cloud의 외부 다운로드 연결을 확인하세요."
            ) from exc

    # MediaPipe의 기본 모델 경로 대신 /tmp에 저장한 모델을 사용한다.
    # MediaPipe 패키지와 네이티브 확장 모듈은 원래 설치 위치에서 그대로 로드된다.
    if hasattr(mp.solutions.pose, "_download_oss_pose_landmark_model"):
        mp.solutions.pose._download_oss_pose_landmark_model = (
            lambda model_complexity: str(model_path)
        )


prepare_writable_pose_model()
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="AI 맞춤형 보조 로봇팔", page_icon="🦾", layout="wide")
st.title("🦾 AI 기반 인체 맞춤형 웨어러블 로봇")
st.caption("사진을 업로드하고 사진 속 기준 길이를 입력하면 AI가 팔 관절을 인식해 실제 길이를 추정하고, 두 종류의 가상 웨어러블 로봇팔 크기에 반영합니다.")

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
    "웹캠 대신 팔 사진을 분석합니다. 사진 속 자의 실제 길이와 픽셀 길이를 이용해 크기를 보정합니다. "
    "자와 팔은 같은 평면에 놓고, 어깨부터 손목까지 팔 전체가 보이도록 촬영하세요. 결과는 가상 시뮬레이션용 추정치입니다."
)

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


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


with st.sidebar:
    st.header("⚙️ 측정 및 로봇 설정")
    st.caption("현재 선택: " + ("소프트 유연형" if robot_type == "soft" else "웨어러블 엑스트라 암"))
    reference_cm = st.number_input(
        "사진 속 기준자의 실제 길이 (cm)", min_value=1.0, max_value=100.0,
        value=30.0, step=0.5,
        help="사진에 함께 찍힌 자의 실제 길이를 입력하세요. 예: 30cm 자라면 30 입력"
    )
    reference_px = st.number_input(
        "사진 속 기준자의 픽셀 길이 (px)", min_value=1.0, max_value=10000.0,
        value=300.0, step=1.0,
        help="사진에서 기준자 양 끝 사이의 픽셀 거리입니다. 이미지 편집기나 픽셀 측정 도구로 확인해 입력합니다."
    )
    robot_scale = st.slider("로봇 크기 배율", 0.5, 1.8, 1.0, 0.05)
    motion_gain = st.slider("동작 반응 배율", 0.5, 2.0, 1.0, 0.1)
    st.caption("배율은 가상 로봇 시뮬레이션에만 적용됩니다.")

cm_per_pixel = reference_cm / max(reference_px, 1.0)
measurements = {}
left, right = st.columns([1.05, 1])
with left:
    st.subheader("1. 팔 사진 업로드 및 AI 관절 인식")
    uploaded = st.file_uploader("팔과 기준자가 함께 보이는 사진을 선택하세요", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        file_bytes = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image_bgr is None:
            st.error("이미지를 읽지 못했습니다. JPG 또는 PNG 파일을 다시 선택해 주세요.")
        else:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            try:
                with mp_pose.Pose(static_image_mode=True, model_complexity=1,
                                  enable_segmentation=False, min_detection_confidence=0.5) as pose_model:
                    pose_result = pose_model.process(rgb)
                with mp_hands.Hands(static_image_mode=True, max_num_hands=2, model_complexity=1,
                                    min_detection_confidence=0.5) as hands_model:
                    hands_result = hands_model.process(rgb)
                annotated = image_bgr.copy()
                height, width = annotated.shape[:2]
                if pose_result.pose_landmarks:
                    lm = pose_result.pose_landmarks.landmark
                    mp_drawing.draw_landmarks(annotated, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    specs = [(11,13,15,"LEFT ARM"),(12,14,16,"RIGHT ARM")]
                    candidates=[]
                    for si,ei,wi,label in specs:
                        pts=[]
                        for idx, name, color in [(si,"START · 어깨",(0,255,0)),(ei,"ELBOW · 팔꿈치",(0,220,255)),(wi,"END · 손목",(255,180,0))]:
                            p=lm[idx]
                            if p.visibility >= 0.35:
                                xy=(int(p.x*width),int(p.y*height)); pts.append(xy)
                                cv2.circle(annotated,xy,8,color,-1,cv2.LINE_AA)
                                cv2.putText(annotated,name,(max(5,xy[0]+8),max(20,xy[1]-8)),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2,cv2.LINE_AA)
                            else: pts.append(None)
                        if pts[0] and pts[1]: cv2.line(annotated,pts[0],pts[1],(0,220,255),4,cv2.LINE_AA)
                        if pts[1] and pts[2]: cv2.line(annotated,pts[1],pts[2],(255,180,0),4,cv2.LINE_AA)
                        if all(pts):
                            upper_px=float(np.hypot((lm[si].x-lm[ei].x)*width,(lm[si].y-lm[ei].y)*height))
                            fore_px=float(np.hypot((lm[ei].x-lm[wi].x)*width,(lm[ei].y-lm[wi].y)*height))
                            candidates.append((upper_px,fore_px,si,ei,wi))
                    if candidates:
                        # Select the more visible/longer detected arm; average if both sides are visible.
                        measurements["upper_arm_px"]=float(np.mean([v[0] for v in candidates]))
                        measurements["forearm_px"]=float(np.mean([v[1] for v in candidates]))
                        measurements["arm_estimation_mode"]="direct"
                        measurements["arm_error_pct"]=12.0 if len(candidates)==1 else 8.0
                        vals=[]
                        for _,_,si,ei,wi in candidates:
                            a=(lm[si].x,lm[si].y); b=(lm[ei].x,lm[ei].y); c=(lm[wi].x,lm[wi].y)
                            ang=angle_2d(a,b,c)
                            if ang is not None: vals.append(clamp((180-ang)/150,0,1)*150)
                        if vals: measurements["elbow_flexion_deg"]=float(np.mean(vals))
                        status="팔 관절 인식 성공"
                    else:
                        status="어깨·팔꿈치·손목이 모두 보이는 팔을 찾지 못했습니다."
                else:
                    status="팔을 인식하지 못했습니다. 상체와 팔 전체가 보이는 사진을 사용하세요."
                if hands_result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(annotated,hands_result.multi_hand_landmarks[0],mp_hands.HAND_CONNECTIONS)
                    h=hands_result.multi_hand_landmarks[0].landmark
                    measurements["index_finger_px"]=float(np.hypot((h[5].x-h[8].x)*width,(h[5].y-h[8].y)*height))
                    measurements["palm_width_px"]=float(np.hypot((h[5].x-h[17].x)*width,(h[5].y-h[17].y)*height))
                    curls=[]
                    for base,tip,ext,ran in [(2,4,1.25,.55),(5,8,1.45,.65),(9,12,1.45,.65),(13,16,1.4,.6),(17,20,1.35,.55)]:
                        bd=np.hypot(h[base].x-h[0].x,h[base].y-h[0].y)
                        td=np.hypot(h[tip].x-h[0].x,h[tip].y-h[0].y)
                        curls.append(clamp((ext-td/max(bd,1e-6))/ran))
                    measurements["finger_curl"]=float(np.mean(curls))
                st.image(cv2.cvtColor(annotated,cv2.COLOR_BGR2RGB),caption="AI가 인식한 어깨·팔꿈치·손목 위치",use_container_width=True)
                st.write("**인식 상태:** "+status)
                if not measurements.get("upper_arm_px"):
                    st.warning("팔 전체가 선명하게 보이는 사진으로 다시 시도해 주세요.")
            except Exception as exc:
                st.error(f"사진 분석 중 오류가 발생했습니다: {exc}")
    else:
        st.info("사진을 업로드하면 AI가 팔 관절을 인식합니다. 팔과 기준자는 카메라에서 비슷한 거리에 놓아 주세요.")

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
    st.caption("주황색은 사용자 팔, 파란색/굵은 모듈은 가상 로봇팔입니다. 점선은 착용 지지·밴드를 개념적으로 표시합니다. 모델 위에 상완·하완 추정 길이도 표시됩니다. 사진 기반 2D 추정이므로 실제 3D 관절 각도와 다를 수 있습니다.")

st.divider()
st.subheader("4. 로봇팔 동작 정확도 예측")

# 업로드한 사진의 관절 인식 상태를 바탕으로 가상 로봇팔의 예상 추적 정확도를 계산
arm_mode = measurements.get("arm_estimation_mode", "unavailable")
arm_error_pct = measurements.get("arm_error_pct")

if arm_mode != "unavailable" and arm_error_pct is not None:
    predicted_accuracy = max(0.0, min(100.0, 100.0 - float(arm_error_pct)))
    st.metric("예상 로봇팔 동작 정확도", f"{predicted_accuracy:.1f}%")
    st.progress(int(round(predicted_accuracy)))

    if predicted_accuracy >= 90:
        st.success("관절 인식 상태가 양호해 로봇팔의 동작 추적 정확도가 높을 것으로 예상됩니다.")
    elif predicted_accuracy >= 80:
        st.info("일부 오차가 예상됩니다. 팔 전체가 카메라에 잘 보이도록 유지하면 정확도를 높일 수 있습니다.")
    else:
        st.warning("관절 가림 또는 추정값 사용으로 오차가 클 수 있습니다. 카메라 위치와 팔 자세를 조정해 주세요.")
else:
    st.warning("팔 관절이 충분히 인식되지 않아 정확도를 예측할 수 없습니다. 어깨부터 손목까지 사진에 보이도록 해 주세요.")

st.caption("예측값은 사진 속 관절 인식 오차를 바탕으로 계산한 참고용 가상 시뮬레이션 지표이며, 실제 제작된 로봇의 기계적 정확도를 측정한 값은 아닙니다. 기준자의 픽셀 길이 입력이 부정확하거나 자와 팔의 깊이가 다르면 실제 치수 오차가 커집니다.")
