# 트러블슈팅 가이드

이 문서는 LiteRT 및 Play Services TFLite 통합 과정에서 발생한 주요 문제와 해결 상태를 정리합니다.

## 1. Play Services TFLite: Native initialization method not found

### 문제 현상
`PlayServicesRunner`에서 `TfLite.initialize(context)` 호출 후 `InterpreterApi.create()` 단계에서 다음과 같은 에러 발생:
`TfLite initialization failed: java.lang.Exception: Native initialization method not found`

### 원인 추정
- **의존성 충돌**: `com.google.ai.edge.litert:litert` 라이브러리와 `org.tensorflow:tensorflow-lite-api` 간의 클래스 중복.
- **Native Library 누락**: 중복 해결을 위해 `build.gradle.kts`에서 `exclude`를 과도하게 적용할 경우, Play Services가 요구하는 JNI 라이브러리(`libtensorflowlite_jni.so`)가 APK에서 누락되거나 잘못된 버전이 포함될 수 있음.

### 해결 시도
- `exclude` 대신 `packagingOptions`의 `pickFirst`를 사용하여 APK 내에 필요한 Native Library가 포함되도록 유도함.
- `org.tensorflow:tensorflow-lite-support` 등 부가 라이브러리에서 발생하는 중복 클래스들을 `pickFirst`로 처리.

---

## 2. LiteRT CompiledModel: Shape Mismatch (1001 vs 1000)

### 문제 현상
`CompiledModelRunner`에서 추론 시 `IllegalArgumentException: The size of the array to be loaded does not match the specified shape.` 에러 발생.
로그 확인 결과: `Result: 1000, FlatSize: 1001`

### 원인
- `ImageClassifier`에서 기본 `mobilenet_v2.tflite` 모델의 출력 크기를 1001(Background class 포함)로 가정하고 버퍼를 생성했으나, LiteRT로 변환된 모델이나 특정 런타임에서는 1000개 클래스만 반환함.

### 해결 방법
- `CompiledModelRunner.classify()` 단계에서 결과 배열의 크기를 체크하여, `supportOutputBuffer`의 크기와 일치하지 않을 경우 동적으로 버퍼를 재할당(`Re-allocating`) 하도록 수정함.

---

## 3. LiteRT API: Kotlin Reflection 필요성

### 문제 현상
LiteRT 2.1.0의 `CompiledModel` 및 `TensorBuffer` API가 기존 TFLite Java API와 메서드 이름이 겹치거나(Kotlin extension), 접근 제어자 문제로 직접 호출 시 컴파일 에러 또는 런타임 `NoSuchMethodException` 발생.

### 해결 방법
- `java.lang.reflect`를 사용하여 런타임에 메서드를 탐색하고 호출하는 방식 채택.
- `TensorBuffer`의 경우 `load(ByteBuffer)` 대신 `writeFloat(float[])` / `readFloat()` 등을 사용하여 데이터를 주고받도록 구현.
