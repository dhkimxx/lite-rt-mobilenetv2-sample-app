# Android TensorFlow Lite Image Classifier

이 프로젝트는 Android CameraX와 TensorFlow Lite를 사용하여 실시간 이미지 분류(Image Classification)를 수행하는 애플리케이션입니다. **MobileNet V2** 모델을 기반으로 하며, 다양한 **하드웨어 가속(CPU/GPU/NPU)** 옵션을 지원합니다.

## 📱 주요 기능

*   **다양한 가속기 지원:**
    *   **CPU (Default):** XNNPACK 위임자를 사용하여 안정적인 성능을 제공합니다.
    *   **GPU:** OpenGL/OpenCL 기반의 하드웨어 가속을 통해 고속 추론을 지원합니다.
    *   **NPU:** Android NNAPI를 활용하여 신경망 처리 장치 가속을 지원합니다.
*   **실시간 성능 모니터링:** 
    *   현재 프레임의 추론 시간(ms) 표시.
    *   **10초 이동 평균(Moving Average)**을 통해 안정적인 성능 지표 제공.
*   **모델 호환성:** FP32(정밀도) 및 INT8(경량화) 모델 간 즉시 전환 가능.
*   **유연한 입력 처리:** 모델의 입력 포맷(NCHW/NHWC)을 자동으로 감지하고, 필요한 경우 실시간으로 데이터를 변환(Permutation)합니다.
*   **갤러리 이미지 분석:** 저장된 사진을 불러와 분석할 수 있습니다.

## 🛠 기술 스택

*   **Language:** Kotlin
*   **UI Framework:** Jetpack Compose (Material3)
*   **ML Engine:** TensorFlow Lite Native (Interpreter API)
    *   `tensorflow-lite` (2.14.0)
    *   `tensorflow-lite-gpu` & `tensorflow-lite-gpu-api`
*   **Camera:** CameraX
*   **Design Pattern:** Thread-safe Repository Pattern (Synchronized ImageClassifier)

## 🔧 주요 구현 사항 및 트러블슈팅

### 1. 하드웨어 가속 및 스레드 안정성
*   **Multi-Delegate Support:** 단일 `Interpreter` 인스턴스에서 옵션 변경 시 `GpuDelegate` 또는 `NnApiDelegate`를 동적으로 적용합니다.
*   **Thread Safety:** 추론(Background)과 모델 교체(UI Thread) 간의 충돌을 방지하기 위해 `ImageClassifier` 전반에 `synchronized` 동기화를 적용하여 앱 안정성을 확보했습니다.

### 2. NCHW 모델 지원 (Custom Permutation)
*   **문제:** Android Bitmap은 NHWC(Pixel Interleaved) 방식이지만, 사용된 MobileNet V2 모델은 NCHW(Channel First) 입력을 요구합니다.
*   **해결:** `ImageUtils`를 통해 Bitmap 데이터를 수동으로 재배열(Permutation)하여 GPU/CPU 모드 관계없이 올바른 입력 텐서를 주입하도록 구현했습니다.

### 3. 입력 정규화 및 라벨 매핑
*   **Normalization:** 입력 픽셀 값 `[0, 255]`를 모델이 요구하는 `[-1, 1]` 범위로 정규화(`NormalizeOp(127.5, 127.5)`)하여 정확도를 개선했습니다.
*   **Label Adjustment:** 1001개 클래스를 출력하는 모델과 1000개 라벨 파일 간의 인덱스 불일치(Background 클래스 0번)를 자동으로 보정합니다.

### 4. 실시간 평균 추론 시간
*   최근 10초간의 추론 이력을 롤링 윈도우(Rolling Window)로 관리하며, 옵션 변경 시 즉시 초기화되어 정확한 성능 비교(Benchmark)가 가능합니다.

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

*   `MainActivity.kt`: UI(Compose) 구성, 권한 처리, 하드웨어 가속 선택 및 성능 통계 표시.
*   `ImageClassifier.kt`: TFLite Interpreter 래퍼. 모델 로드, 가속기 설정(Delegates), 이미지 전처리, 스레드 동기화 담당.
*   `assets/`: 
    *   `mobilenet_v2.tflite` (FP32, NCHW)
    *   `mobilenet_v2_aiedge_int8.tflite` (INT8, NCHW)
    *   `labels.txt`

---
**Note:** GPU 가속은 기기 호환성에 따라 지원되지 않을 수 있으며, 이 경우 로그에 Warning이 출력되고 CPU 모드로 동작할 수 있습니다.
