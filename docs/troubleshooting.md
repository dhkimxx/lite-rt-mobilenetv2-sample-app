# 트러블슈팅 가이드 (Troubleshooting Guide)

이 문서는 상용 수준의 Android TFLite 이미지 분류기를 구현하며 겪은 기술적 난제들과 그 해결 과정을 기록합니다.

## 1. 라이브러리 의존성 및 Native Crash
### 문제 현상
- TFLite 가속Delegate(GPU/NNAPI) 사용 시 `NoClassDefFoundError` 또는 `Native initialization method not found` 에러 발생.
- `tensorflow-lite-gpu` 최신 버전(2.16+)에서 JNI 구조가 변경되어 기존 API와 호환되지 않는 문제.

### 해결 과정
- **버전 고정**: TFLite 버전을 안정적인 **2.14.0**으로 고정.
- **명시적 의존성**: `tensorflow-lite-gpu-api`를 별도로 추가하여 Delegate 인터페이스 중복 문제를 해결.
- **LiteRT 제거**: 과도하게 복잡한 Google AI Edge(LiteRT) 라이브러리를 제거하고 표준 TFLite Interpreter로 회귀하여 안정성을 확보함.

---

## 2. NCHW 모델 지원 및 데이터 정렬 문제
### 문제 현상
- MobileNet V2 모델이 NHWC(일반)가 아닌 **NCHW(Planar)** 입력을 요구하면서 추론 결과가 특정 라벨로 고정되거나 정확도가 2~5% 수준으로 급락함.
- `ImageProcessor`를 통한 자동 변환 시 버퍼 정렬(Alignment) 문제로 데이터가 오염됨.

### 해결 과정
- **수동 픽셀 추출**: `Bitmap.getPixels()`를 통해 픽셀 로우 데이터를 직접 추출.
- **커스텀 Permutation**: `R-G-B`가 섞여 있는 데이터를 `RRR...GGG...BBB...` 순으로 재배열하는 로직을 직접 구현.
- **이미지 정규화**: FP32 모델의 경우 `(Pixel - 127.5) / 127.5` 식을 적용해 `[-1, 1]` 범위의 부동 소수점으로 정밀 주입.

---

## 3. INT8 양자화 모델의 확률 왜곡
### 문제 현상
- INT8 모델 추론 시 결과 확률이 모두 0%로 나오거나, Top 3가 동일한 확률로 표시됨.

### 해결 과정
- **역양자화(Dequantization)**: 모델의 출력 텐서에서 `QuantizationParams` (Scale, ZeroPoint)를 추출.
- **로직 수정**: `(Output - ZeroPoint) * Scale` 공식을 적용해 정수형 출력을 실수형 Logits으로 복원한 후 Softmax를 적용하도록 개선.

---

## 4. 실시간 가속기 전환 및 스레드 안정성
### 문제 현상
- 카메라 추론 중(Background) 모델이나 하드웨어 가속기(CPU/GPU/NPU)를 변경(Safe/Main)하면 Native Race Condition으로 인해 앱이 즉시 종료됨.

### 해결 과정
- **동기화 잠금**: `synchronized(lock)` 블록을 `classify()`와 `initialize()` 전반에 적용하여 자원 교체 중에 추론이 실행되지 않도록 강제함.
- **자원 해제**: Delegate 교체 시 기존 자원을 `close()`를 통해 명시적으로 해제하여 메모리 누수 방지.

---

## 5. 지속되는 과제: INT8 모델 신뢰도 저하
### 현재 상태
- FP32 모델은 90% 이상의 신뢰도를 보이나, INT8 모델은 동일 환경에서 한자리 수(5~9%)의 신뢰도를 보이는 현상 잔존.

### 분석 결과 (로그 분석)
- **Logits Saturation**: INT8 모델의 출력 로그 확인 결과, 상위 5~10개 이상의 라벨이 동시에 **Raw UINT8 값 255(최댓값)**에 도달하여 포화(Saturation)되는 현상 발견.
- **원인 추정**: 
    1. **BGR 채널 반전**: 모델이 BGR 입력을 기대하는데 RGB를 주입할 경우, 색상 필터 오작동으로 인해 전체적인 활성화가 비정상적으로 높아짐.
    2. **입력 오프셋(Mean/Std) 불일치**: 모델이 `[0, 1]` 범위를 기대하나 실제 데이터의 편향(Bias)이 제거되지 않아 뉴런들이 과도하게 흥분됨.

### 해결 시도
- **BGR 수동 변환**: 입력 데이터의 R과 B 채널을 교체하여 주입한 후 결과 관찰.
- **Top-1 확률 복구**: 포화 문제가 해결되어 로그값이 분산되면 Softmax 결과가 90% 이상으로 회복될 것으로 기대.
