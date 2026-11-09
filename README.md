# AI 기반 인체 맞춤형 웨어러블 로봇팔

## 실행
```bash
streamlit run main.py
```

## Streamlit Community Cloud 배포
1. 이 폴더 안의 파일을 GitHub 저장소 루트에 업로드합니다.
2. Streamlit Cloud에서 `main.py`를 앱 시작 파일로 지정합니다.
3. Python 3.11 환경을 사용합니다.

웹캠은 HTTPS 환경과 브라우저 카메라 권한이 필요합니다.

### MediaPipe 모델 파일 처리
Streamlit Cloud의 `site-packages` 경로에 쓰기 권한이 없어 Pose 모델 자동 다운로드가 실패하는 경우를 해결하기 위해, 앱 시작 시 MediaPipe 패키지를 쓰기 가능한 `/tmp` 경로로 복사하고 Pose 모델도 해당 경로에 준비합니다. 따라서 GitHub에 별도의 모델 파일을 올릴 필요가 없습니다. 최초 실행 시 모델 다운로드를 위해 Streamlit Cloud의 외부 네트워크 연결이 필요합니다.
