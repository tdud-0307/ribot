# AI 기반 인체 맞춤형 웨어러블 로봇팔

## GitHub 저장소에 올리는 방법

압축을 푼 뒤 **폴더 안의 파일 5개를 GitHub 저장소 최상위(root)에 직접 업로드**하세요.

저장소 최상위가 아래처럼 보여야 합니다.

```text
ribot/
├── main.py
├── requirements.txt
├── packages.txt
├── runtime.txt
└── README.md
```

`robot_arm_cloud_python311_fix` 같은 바깥 폴더째 업로드하면 안 됩니다. `main.py`가 저장소 최상위에 있어야 합니다.

## Streamlit Community Cloud 설정

- Main file path: `main.py`
- Python version: **3.11**

중요: 배포 로그에 `Using Python 3.13`이라고 계속 나오면 현재 앱이 Python 3.13 환경으로 만들어진 것입니다. `runtime.txt`만으로 기존 앱의 실행 환경이 바뀌지 않을 수 있으므로, 앱 설정에서 Python 3.11을 선택할 수 있는지 확인하세요. 변경할 수 없다면 Python 3.11을 선택해 새 앱으로 배포하고 같은 GitHub 저장소와 `main.py`를 지정하세요.

## 포함 파일

- `main.py`: 웹캠 기반 팔·손 인식과 3D 웨어러블 로봇팔 시각화
- `requirements.txt`: Python 3.11과 호환되는 패키지 버전 고정
- `packages.txt`: 시스템 라이브러리
- `runtime.txt`: Python 3.11 지정

웹캠은 HTTPS 환경과 브라우저 카메라 권한이 필요합니다. 앱은 가상 시뮬레이션이며 실제 로봇 하드웨어를 제어하지 않습니다.
