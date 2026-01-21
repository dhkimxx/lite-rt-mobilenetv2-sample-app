# Android TensorFlow Lite Image Classifier

이 프로젝트는 Android CameraX와 TensorFlow Lite를 사용하여 실시간 이미지 분류(Image Classification)를 수행하는 애플리케이션입니다. **MobileNet V2** 모델을 사용하여 사물을 인식하며, 갤러리 이미지 선택 기능도 지원합니다.

## 📱 주요 기능

*   **실시간 카메라 추론:** CameraX를 통해 입력받은 프레임에 대해 실시간으로 사물을 분류합니다.
*   **갤러리 이미지 분석:** 저장된 사진을 불러와 분석할 수 있습니다.
*   **정확한 확률 표시:** Softmax 알고리즘을 적용하여 0~100%의 정확한 신뢰도를 표시합니다.
*   **성능 모니터링:** 추론에 소요되는 시간(Inference Time)을 밀리초(ms) 단위로 표시합니다.

## 🛠 기술 스택

*   **Language:** Kotlin
*   **UI Framework:** Jetpack Compose (Material3)
*   **ML Library:** TensorFlow Lite (with Support Library)
*   **Camera:** CameraX
*   **Build System:** Gradle (Kotlin DSL)

## 🔧 주요 구현 사항 및 트러블슈팅

이 프로젝트는 개발 과정에서 다음과 같은 기술적 이슈들을 해결하여 완성도를 높였습니다.

### 1. 데이터 포맷 변환 (Transposition)
*   **문제:** Android Bitmap 및 Buffer는 `NHWC` (Interleaved RGB) 형식을 사용하지만, 학습된 MobileNet V2 모델은 `NCHW` (Planar RGB) 형식을 요구하여 추론 정확도가 매우 낮았습니다.
*   **해결:** `ImageClassifier.kt`에 **NHWC → NCHW 변환 로직**을 구현하여 입력 텐서의 차원을 재배열하였습니다. 이를 통해 모델이 이미지를 올바르게 인식하도록 수정했습니다.

### 2. 확률 정규화 (Softmax)
*   **문제:** 모델의 출력값이 정규화되지 않은 Logit(Raw Score) 형태여서, 300%와 같은 비정상적인 수치가 표시되었습니다.
*   **해결:** 출력 버퍼에 **Softmax 함수**를 적용하여 모든 클래스의 확률 합이 1(100%)이 되도록 정규화했습니다.

### 3. 빌드 및 호환성 해결
*   AGP(Android Gradle Plugin)와 Kotlin 버전 간의 호환성 문제를 해결하기 위해 버전을 조정하고, 누락된 `kotlin-android` 플러그인을 복구했습니다.

## 🚀 설치 및 실행 방법

1.  **Clone Project:**
    ```bash
    git clone <repository-url>
    ```
2.  **Open in Android Studio:** 프로젝트 폴더를 엽니다.
3.  **Build & Run:**
    *   USB 디버깅이 활성화된 Android 기기를 연결합니다.
    *   `Run` 버튼을 누르거나 터미널에서 아래 명령어를 실행합니다.
    ```bash
    ./gradlew installDebug
    ```

## 📂 프로젝트 구조

*   `MainActivity.kt`: UI(Compose) 구성 및 권한 처리.
*   `ImageClassifier.kt`: TFLite 모델 로드, 전처리(Preprocessing), 후처리(Postprocessing: Softmax), 추론 실행 담당.
*   `assets/`: `mobilenet_v2.tflite` 모델 파일과 `labels.txt` 파일 포함.

---
**Note:** 이 앱은 `labels.txt`에 정의된 1,000개의 ImageNet 클래스에 대해 분류를 수행합니다.
