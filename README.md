# Android TensorFlow Lite Image Classifier

이 프로젝트는 Android CameraX와 TensorFlow Lite를 사용하여 실시간 이미지 분류(Image Classification)를 수행하는 애플리케이션입니다. **MobileNet V2** 모델(FP32 및 INT8)을 사용하여 사물을 인식하며, 실시간 모델 교체 기능을 지원합니다.

## 📱 주요 기능

*   **실시간 카메라 추론:** CameraX를 통해 입력받은 프레임에 대해 실시간으로 사물을 분류합니다.
*   **모델 선택 (FP32 vs INT8):** 실행 중 정밀도(FP32)와 경량화(INT8) 모델을 즉시 전환하여 비교할 수 있습니다.
*   **갤러리 이미지 분석:** 저장된 사진을 불러와 분석할 수 있습니다.
*   **정확한 확률 표시:** Softmax 알고리즘을 적용하여 0~100%의 정확한 신뢰도를 표시합니다.
*   **성능 모니터링:** 추론에 소요되는 시간(Inference Time)을 밀리초(ms) 단위로 표시합니다.

## 🛠 기술 스택

*   **Language:** Kotlin
*   **UI Framework:** Jetpack Compose (Material3)
*   **ML Library:** TensorFlow Lite (with Support Library)
*   **Camera:** CameraX
*   **Concurrency:** Kotlin Coroutines & Executors
*   **Build System:** Gradle (Kotlin DSL)

## 🔧 주요 구현 사항 및 트러블슈팅

이 프로젝트는 개발 과정에서 다음과 같은 기술적 이슈들을 해결하여 완성도를 높였습니다.

### 1. 동적 모델 교체 및 스레드 안전성
*   **기능:** 사용자가 UI에서 FP32와 INT8 모델을 선택하면 `ImageClassifier`가 즉시 모델을 다시 로드합니다.
*   **문제 해결:** 추론(Background Thread)이 진행되는 도중 모델을 교체(Main Thread)할 경우 발생하는 Native Crash(Race Condition)를 방지하기 위해, `synchronized` 블록을 사용하여 **Thread Safety**를 확보했습니다.

### 2. INT8 양자화 모델 지원 (UINT8 NCHW)
*   **문제:** 사용된 INT8 모델이 일반적인 NHWC 형식이 아닌 `NCHW` 형식을 사용하고 있어, 단순 버퍼 복사 시 결과가 왜곡되었습니다.
*   **해결:** `ImageClassifier.kt`에 **Uint8 데이터에 대한 NCHW 변환(Byte Permutation)** 로직을 추가하여, 포맷 불일치로 인한 정확도 저하 문제를 완전히 해결했습니다.

### 3. 확률 정규화 (Softmax)
*   **문제:** 모델의 출력값이 정규화되지 않은 Logit(Raw Score) 형태여서, 300%와 같은 비정상적인 수치가 표시되었습니다.
*   **해결:** 출력 버퍼에 **Softmax 함수**를 적용하여 모든 클래스의 확률 합이 1(100%)이 되도록 정규화했습니다.

### 4. 데이터 포맷 변환 (Transposition)
*   FP32 모델 또한 NCHW 입력을 요구하므로, Android Bitmap(NHWC)을 모델에 맞는 Planar 포맷으로 변환하는 전처리 로직을 구현했습니다.

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

*   `MainActivity.kt`: UI(Compose) 구성, 권한 처리, 모델 선택 상태 관리.
*   `ImageClassifier.kt`: TFLite 모델 로드, 전처리(Resize/Permutation), 후처리(Softmax), 동기화된 추론 실행 담당.
*   `assets/`: `mobilenet_v2.tflite` (FP32), `mobilenet_v2_aiedge_int8.tflite` (INT8), `labels.txt` 포함.

---
**Note:** 이 앱은 `labels.txt`에 정의된 1,000개의 ImageNet 클래스에 대해 분류를 수행합니다.
