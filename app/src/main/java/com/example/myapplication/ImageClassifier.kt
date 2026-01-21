package com.example.myapplication

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import kotlin.math.exp

/**
 * NCHW 변환 및 버퍼 관리를 위한 유틸리티 객체
 * DRY 원칙 준수를 위해 공통 로직 분리
 */
object ImageUtils {
    
    /**
     * NHWC -> NCHW 변환 (Float32)
     * @param nhwcBuffer NHWC 포맷의 소스 버퍼
     * @param height 이미지 높이
     * @param width 이미지 너비
     * @param destBuffer 재사용할 대상 버퍼 (null이면 새로 생성)
     * @return NCHW 포맷으로 변환된 ByteBuffer
     */
    fun permuteNHWCToNCHW_Float(
        nhwcBuffer: ByteBuffer,
        height: Int,
        width: Int,
        destBuffer: ByteBuffer? = null
    ): ByteBuffer {
        nhwcBuffer.rewind()
        val floatBuffer = nhwcBuffer.asFloatBuffer()
        val nhwcArray = FloatArray(floatBuffer.remaining())
        floatBuffer.get(nhwcArray)
        
        val nchwArray = FloatArray(nhwcArray.size)
        for (h in 0 until height) {
            for (w in 0 until width) {
                val pixelIndex = (h * width) + w
                nchwArray[pixelIndex] = nhwcArray[pixelIndex * 3 + 0]             // R
                nchwArray[(height * width) + pixelIndex] = nhwcArray[pixelIndex * 3 + 1]  // G
                nchwArray[(2 * height * width) + pixelIndex] = nhwcArray[pixelIndex * 3 + 2] // B
            }
        }
        
        val outputBuffer = destBuffer ?: ByteBuffer.allocateDirect(nchwArray.size * 4)
        outputBuffer.order(ByteOrder.nativeOrder())
        outputBuffer.rewind()
        outputBuffer.asFloatBuffer().put(nchwArray)
        outputBuffer.rewind()
        return outputBuffer
    }
    
    /**
     * NHWC -> NCHW 변환 (UINT8)
     * @param nhwcBuffer NHWC 포맷의 소스 버퍼
     * @param height 이미지 높이
     * @param width 이미지 너비
     * @param destBuffer 재사용할 대상 버퍼 (null이면 새로 생성)
     * @return NCHW 포맷으로 변환된 ByteBuffer
     */
    fun permuteNHWCToNCHW_Uint8(
        nhwcBuffer: ByteBuffer,
        height: Int,
        width: Int,
        destBuffer: ByteBuffer? = null
    ): ByteBuffer {
        nhwcBuffer.rewind()
        val nhwcArray = ByteArray(nhwcBuffer.remaining())
        nhwcBuffer.get(nhwcArray)
        
        val nchwArray = ByteArray(nhwcArray.size)
        for (h in 0 until height) {
            for (w in 0 until width) {
                val pixelIndex = (h * width) + w
                nchwArray[pixelIndex] = nhwcArray[pixelIndex * 3 + 0]             // R
                nchwArray[(height * width) + pixelIndex] = nhwcArray[pixelIndex * 3 + 1]  // G
                nchwArray[(2 * height * width) + pixelIndex] = nhwcArray[pixelIndex * 3 + 2] // B
            }
        }
        
        val outputBuffer = destBuffer ?: ByteBuffer.allocateDirect(nchwArray.size)
        outputBuffer.order(ByteOrder.nativeOrder())
        outputBuffer.rewind()
        outputBuffer.put(nchwArray)
        outputBuffer.rewind()
        return outputBuffer
    }
    
    /**
     * 입력 Shape가 NCHW 포맷인지 확인
     */
    fun isNCHW(shape: IntArray): Boolean {
        return shape.size == 4 && shape[1] == 3
    }
}

/**
 * ModelRunner 인터페이스
 * 전략 패턴: 다양한 추론 런타임을 추상화
 */
interface ModelRunner {
    fun load(modelName: String)
    fun classify(
        tensorImage: TensorImage,
        imageProcessor: ImageProcessor
    ): Pair<TensorBuffer, Long>
    fun getInputDataType(): DataType
    fun getOutputDataType(): DataType
    fun getInputShape(): IntArray
    fun getOutputShape(): IntArray
    fun close()
}

/**
 * 1. 앱 내장 TFLite Interpreter 구현체
 * 버퍼 재사용 최적화 적용
 */
class InterpreterRunner(private val context: Context) : ModelRunner {
    private var interpreter: Interpreter? = null
    
    // 버퍼 재사용을 위한 멤버 변수 (load 시점에 할당)
    private var inputBuffer: ByteBuffer? = null
    private var outputBuffer: TensorBuffer? = null
    private var cachedInputShape: IntArray = intArrayOf(1, 224, 224, 3)
    private var cachedOutputShape: IntArray = intArrayOf(1, 1001)
    private var isNCHW: Boolean = false
    
    override fun load(modelName: String) {
        close()
        try {
            val model: MappedByteBuffer = FileUtil.loadMappedFile(context, modelName)
            val options = Interpreter.Options()
            interpreter = Interpreter(model, options)
            
            // 버퍼 사전 할당
            cachedInputShape = interpreter?.getInputTensor(0)?.shape() ?: intArrayOf(1, 224, 224, 3)
            cachedOutputShape = interpreter?.getOutputTensor(0)?.shape() ?: intArrayOf(1, 1001)
            isNCHW = ImageUtils.isNCHW(cachedInputShape)
            
            val inputDataType = getInputDataType()
            val inputSize = cachedInputShape.reduce { acc, i -> acc * i }
            val bytesPerElement = if (inputDataType == DataType.FLOAT32) 4 else 1
            inputBuffer = ByteBuffer.allocateDirect(inputSize * bytesPerElement).order(ByteOrder.nativeOrder())
            
            outputBuffer = TensorBuffer.createFixedSize(cachedOutputShape, getOutputDataType())
            
            android.util.Log.d("InterpreterRunner", "Loaded $modelName, NCHW: $isNCHW, Input: $inputDataType")
        } catch (e: Exception) {
            e.printStackTrace()
            android.util.Log.e("InterpreterRunner", "Error loading model: ${e.message}")
        }
    }

    override fun classify(
        tensorImage: TensorImage, 
        imageProcessor: ImageProcessor
    ): Pair<TensorBuffer, Long> {
        if (interpreter == null || inputBuffer == null || outputBuffer == null) {
            return Pair(TensorBuffer.createFixedSize(cachedOutputShape, DataType.FLOAT32), 0L)
        }
        
        val processedImage = imageProcessor.process(tensorImage)
        outputBuffer!!.buffer.rewind()
        
        val finalInputBuffer: ByteBuffer = if (isNCHW) {
            val height = cachedInputShape[2]
            val width = cachedInputShape[3]
            if (getInputDataType() == DataType.FLOAT32) {
                ImageUtils.permuteNHWCToNCHW_Float(processedImage.buffer, height, width, inputBuffer)
            } else {
                ImageUtils.permuteNHWCToNCHW_Uint8(processedImage.buffer, height, width, inputBuffer)
            }
        } else {
            processedImage.buffer.also { it.rewind() }
        }
        
        val startTime = android.os.SystemClock.uptimeMillis()
        interpreter?.run(finalInputBuffer, outputBuffer!!.buffer)
        val inferenceTime = android.os.SystemClock.uptimeMillis() - startTime
        
        return Pair(outputBuffer!!, inferenceTime)
    }

    override fun getInputDataType(): DataType = interpreter?.getInputTensor(0)?.dataType() ?: DataType.FLOAT32
    override fun getOutputDataType(): DataType = interpreter?.getOutputTensor(0)?.dataType() ?: DataType.FLOAT32
    override fun getInputShape(): IntArray = cachedInputShape
    override fun getOutputShape(): IntArray = cachedOutputShape

    override fun close() {
        interpreter?.close()
        interpreter = null
        inputBuffer = null
        outputBuffer = null
    }
}

/**
 * 2. Google Play Services TFLite Runtime 구현체
 * InterpreterApi + TfLiteRuntime.FROM_SYSTEM_ONLY 사용
 */
class PlayServicesRunner(private val context: Context) : ModelRunner {
    private var interpreter: org.tensorflow.lite.InterpreterApi? = null
    private var modelBuffer: MappedByteBuffer? = null
    
    // 버퍼 재사용을 위한 멤버 변수
    private var inputBuffer: ByteBuffer? = null
    private var outputBuffer: TensorBuffer? = null
    private var cachedInputShape: IntArray = intArrayOf(1, 224, 224, 3)
    private var cachedOutputShape: IntArray = intArrayOf(1, 1001)
    private var isNCHW: Boolean = false
    private var pendingModelName: String? = null
    
    override fun load(modelName: String) {
        close()
        pendingModelName = modelName
        try {
            // Play Services TFLite 초기화 (비동기, 필수!)
            com.google.android.gms.tflite.java.TfLite.initialize(context).addOnSuccessListener {
                android.util.Log.d("PlayServicesRunner", "TfLite initialized successfully")
                loadModelAfterInit(modelName)
            }.addOnFailureListener { e ->
                android.util.Log.e("PlayServicesRunner", "TfLite initialization failed: ${e.toString()}")
            }
        } catch (e: Exception) {
            e.printStackTrace()
            android.util.Log.e("PlayServicesRunner", "Error initializing TfLite: ${e.message}")
        }
    }
    
    private fun loadModelAfterInit(modelName: String) {
        try {
            modelBuffer = FileUtil.loadMappedFile(context, modelName)
            
            val options = org.tensorflow.lite.InterpreterApi.Options()
                // .setRuntime(org.tensorflow.lite.InterpreterApi.Options.TfLiteRuntime.FROM_SYSTEM_ONLY)
            
            interpreter = org.tensorflow.lite.InterpreterApi.create(modelBuffer!!, options)
            
            // 버퍼 사전 할당
            cachedInputShape = interpreter?.getInputTensor(0)?.shape() ?: intArrayOf(1, 224, 224, 3)
            cachedOutputShape = interpreter?.getOutputTensor(0)?.shape() ?: intArrayOf(1, 1001)
            isNCHW = ImageUtils.isNCHW(cachedInputShape)
            
            val inputDataType = getInputDataType()
            val inputSize = cachedInputShape.reduce { acc, i -> acc * i }
            val bytesPerElement = if (inputDataType == DataType.FLOAT32) 4 else 1
            inputBuffer = ByteBuffer.allocateDirect(inputSize * bytesPerElement).order(ByteOrder.nativeOrder())
            
            outputBuffer = TensorBuffer.createFixedSize(cachedOutputShape, getOutputDataType())
            
            android.util.Log.d("PlayServicesRunner", "Loaded $modelName via Play Services, NCHW: $isNCHW, Input: $inputDataType")
        } catch (e: Exception) {
            e.printStackTrace()
            android.util.Log.e("PlayServicesRunner", "Error creating interpreter: ${e.message}")
        }
    }

    override fun classify(
        tensorImage: TensorImage,
        imageProcessor: ImageProcessor
    ): Pair<TensorBuffer, Long> {
        if (interpreter == null || inputBuffer == null || outputBuffer == null) {
            return Pair(TensorBuffer.createFixedSize(cachedOutputShape, DataType.FLOAT32), 0L)
        }
        
        val processedImage = imageProcessor.process(tensorImage)
        outputBuffer!!.buffer.rewind()
        
        val finalInputBuffer: ByteBuffer = if (isNCHW) {
            val height = cachedInputShape[2]
            val width = cachedInputShape[3]
            if (getInputDataType() == DataType.FLOAT32) {
                ImageUtils.permuteNHWCToNCHW_Float(processedImage.buffer, height, width, inputBuffer)
            } else {
                ImageUtils.permuteNHWCToNCHW_Uint8(processedImage.buffer, height, width, inputBuffer)
            }
        } else {
            processedImage.buffer.also { it.rewind() }
        }
        
        val startTime = android.os.SystemClock.uptimeMillis()
        interpreter?.run(finalInputBuffer, outputBuffer!!.buffer)
        val inferenceTime = android.os.SystemClock.uptimeMillis() - startTime
        
        return Pair(outputBuffer!!, inferenceTime)
    }

    override fun getInputDataType(): DataType = interpreter?.getInputTensor(0)?.dataType() ?: DataType.FLOAT32
    override fun getOutputDataType(): DataType = interpreter?.getOutputTensor(0)?.dataType() ?: DataType.FLOAT32
    override fun getInputShape(): IntArray = cachedInputShape
    override fun getOutputShape(): IntArray = cachedOutputShape

    override fun close() {
        interpreter?.close()
        interpreter = null
        inputBuffer = null
        outputBuffer = null
    }
}

/**
 * 3. LiteRT CompiledModel Runtime 구현체
 * 최신 LiteRT 2.1.0 CompiledModel API 사용 (Kotlin 2.1.0+ 필요)
 */
class CompiledModelRunner(private val context: Context) : ModelRunner {
    private var compiledModel: com.google.ai.edge.litert.CompiledModel? = null
    private var litertInputBuffers: List<Any>? = null
    private var litertOutputBuffers: List<Any>? = null
    
    // 버퍼 재사용을 위한 멤버 변수
    private var inputBuffer: ByteBuffer? = null
    private var supportOutputBuffer: TensorBuffer? = null // TFLite Support's TensorBuffer
    private var cachedInputShape: IntArray = intArrayOf(1, 224, 224, 3)
    private var cachedOutputShape: IntArray = intArrayOf(1, 1001)
    private var cachedInputDataType: DataType = DataType.FLOAT32
    private var cachedOutputDataType: DataType = DataType.FLOAT32
    private var isNCHW: Boolean = false
    
    override fun load(modelName: String) {
        close()
        try {
            // assets에서 캐시 디렉토리로 모델 복사 (CompiledModel은 파일 경로 필요)
            val cacheFile = java.io.File(context.cacheDir, modelName)
            if (!cacheFile.exists()) {
                context.assets.open(modelName).use { inputStream ->
                    java.io.FileOutputStream(cacheFile).use { outputStream ->
                        inputStream.copyTo(outputStream)
                    }
                }
            }
            
            // CompiledModel 생성 (파일 경로 사용)
            compiledModel = com.google.ai.edge.litert.CompiledModel.create(cacheFile.absolutePath)
            
            // LiteRT TensorBuffers 사전 할당
            compiledModel?.let { model ->
                try {
                    val createInputMethod = model.javaClass.getMethod("createInputBuffers")
                    val createOutputMethod = model.javaClass.getMethod("createOutputBuffers")
                    litertInputBuffers = createInputMethod.invoke(model) as? List<Any>
                    litertOutputBuffers = createOutputMethod.invoke(model) as? List<Any>
                    android.util.Log.d("CompiledModelRunner", "Pre-allocated LiteRT TensorBuffers")
                } catch (e: Exception) {
                    android.util.Log.e("CompiledModelRunner", "Failed to pre-allocate LiteRT buffers: ${e.message}")
                }
            }
            
            // shape와 타입 캐싱 (classify에서 동적 재할당하므로 기본값 설정)
            isNCHW = true
            cachedInputShape = intArrayOf(1, 3, 224, 224)
            cachedOutputShape = intArrayOf(1, 1001)
            cachedInputDataType = if (modelName.contains("int8")) DataType.UINT8 else DataType.FLOAT32
            cachedOutputDataType = DataType.FLOAT32
            
            // 버퍼 사전 할당
            val inputSize = cachedInputShape.reduce { acc, i -> acc * i }
            val bytesPerElement = if (cachedInputDataType == DataType.FLOAT32) 4 else 1
            inputBuffer = ByteBuffer.allocateDirect(inputSize * bytesPerElement).order(ByteOrder.nativeOrder())
            
            supportOutputBuffer = TensorBuffer.createFixedSize(cachedOutputShape, cachedOutputDataType)
            
            android.util.Log.d("CompiledModelRunner", "Loaded $modelName via CompiledModel, NCHW: $isNCHW, Input: $cachedInputDataType, Output: ${cachedOutputShape.contentToString()}")
        } catch (e: Exception) {
            e.printStackTrace()
            android.util.Log.e("CompiledModelRunner", "Error loading CompiledModel: ${e.message}")
        }
    }

    override fun classify(
        tensorImage: TensorImage,
        imageProcessor: ImageProcessor
    ): Pair<TensorBuffer, Long> {
        val model = compiledModel
        val inBuffers = litertInputBuffers
        val outBuffers = litertOutputBuffers
        
        if (model == null || inputBuffer == null || supportOutputBuffer == null || inBuffers == null || outBuffers == null) {
            return Pair(TensorBuffer.createFixedSize(cachedOutputShape, DataType.FLOAT32), 0L)
        }
        
        // 1. 이미지 전처리
        val processedImage = imageProcessor.process(tensorImage)
        
        // 2. NCHW 변환 및 입력 버퍼 로드
        inputBuffer!!.rewind()
        if (isNCHW) {
            val height = cachedInputShape[2]
            val width = cachedInputShape[3]
            if (cachedInputDataType == DataType.FLOAT32) {
                ImageUtils.permuteNHWCToNCHW_Float(processedImage.buffer, height, width, inputBuffer!!)
            } else {
                ImageUtils.permuteNHWCToNCHW_Uint8(processedImage.buffer, height, width, inputBuffer!!)
            }
        } else {
            inputBuffer!!.put(processedImage.buffer)
        }
        
        val startTime = android.os.SystemClock.uptimeMillis()
        
        // 3. LiteRT TensorBuffer에 데이터 로드
        try {
            val inputTensor = inBuffers[0]
            val outputTensor = outBuffers[0]
            
            if (cachedInputDataType == DataType.FLOAT32) {
                val floatArray = FloatArray(inputBuffer!!.remaining() / 4)
                inputBuffer!!.rewind()
                inputBuffer!!.asFloatBuffer().get(floatArray)
                
                val writeMethod = inputTensor.javaClass.getMethod("writeFloat", FloatArray::class.java)
                writeMethod.invoke(inputTensor, floatArray)
            } else {
                val byteArray = ByteArray(inputBuffer!!.remaining())
                inputBuffer!!.rewind()
                inputBuffer!!.get(byteArray)
                
                val writeMethod = inputTensor.javaClass.getMethod("writeInt8", ByteArray::class.java)
                writeMethod.invoke(inputTensor, byteArray)
            }
            
            // 4. 모델 실행 (CompiledModel)
            val runMethod = model.javaClass.getMethod("run", java.util.List::class.java, java.util.List::class.java)
            runMethod.invoke(model, inBuffers, outBuffers)
            
            // 5. 출력 결과 복사
            val readMethod = if (cachedOutputDataType == DataType.FLOAT32) {
                outputTensor.javaClass.getMethod("readFloat")
            } else {
                outputTensor.javaClass.getMethod("readInt8")
            }
            
            val resultArray = readMethod.invoke(outputTensor)
            if (resultArray is FloatArray) {
                if (resultArray.size != supportOutputBuffer!!.flatSize) {
                    android.util.Log.w("CompiledModelRunner", "Output size mismatch. Expected: ${supportOutputBuffer!!.flatSize}, Got: ${resultArray.size}. Re-allocating.")
                    cachedOutputShape = intArrayOf(1, resultArray.size)
                    supportOutputBuffer = TensorBuffer.createFixedSize(cachedOutputShape, cachedOutputDataType)
                }
                supportOutputBuffer!!.loadArray(resultArray)
            } else if (resultArray is ByteArray) {
                if (resultArray.size != supportOutputBuffer!!.flatSize) {
                    android.util.Log.w("CompiledModelRunner", "Output byte size mismatch. Expected: ${supportOutputBuffer!!.flatSize}, Got: ${resultArray.size}. Re-allocating.")
                    cachedOutputShape = intArrayOf(1, resultArray.size)
                    supportOutputBuffer = TensorBuffer.createFixedSize(cachedOutputShape, cachedOutputDataType)
                }
                val floatArray = FloatArray(resultArray.size) { resultArray[it].toFloat() }
                supportOutputBuffer!!.loadArray(floatArray)
            }
            
        } catch (e: Exception) {
            android.util.Log.e("CompiledModelRunner", "Inference failed: ${e.toString()}")
            e.cause?.let { android.util.Log.e("CompiledModelRunner", "Cause: ${it.toString()}") }
        }
        
        val inferenceTime = android.os.SystemClock.uptimeMillis() - startTime
        return Pair(supportOutputBuffer!!, inferenceTime)
    }

    override fun getInputDataType(): DataType = cachedInputDataType
    override fun getOutputDataType(): DataType = cachedOutputDataType
    override fun getInputShape(): IntArray = cachedInputShape
    override fun getOutputShape(): IntArray = cachedOutputShape

    override fun close() {
        try {
            compiledModel?.close()
        } catch (e: Exception) {
            android.util.Log.e("CompiledModelRunner", "Error closing CompiledModel: ${e.message}")
        }
        compiledModel = null
        inputBuffer = null
        supportOutputBuffer = null
        litertInputBuffers = null
        litertOutputBuffers = null
    }
}

/**
 * 이미지 분류 결과
 */
data class InferenceResult(
    val predictions: List<Pair<String, Float>>,
    val inferenceTime: Long
)

/**
 * ImageClassifier - 고수준 API
 * ModelRunner를 사용하여 런타임 독립적으로 동작
 */
class ImageClassifier(private val context: Context) {

    private var labels: List<String> = emptyList()
    private val lock = Any()
    
    enum class Runtime {
        INTERPRETER,      // 앱 내장 TFLite 라이브러리
        PLAY_SERVICES,    // Google Play Services TFLite
        COMPILED_MODEL    // LiteRT CompiledModel API
    }

    private var currentRuntime: Runtime = Runtime.INTERPRETER
    private var modelWrapper: ModelRunner = InterpreterRunner(context)
    
    private var inputImageWidth: Int = 224
    private var inputImageHeight: Int = 224
    private var modelName = "mobilenet_v2.tflite"

    init {
        loadLabels()
        loadModel(modelName, currentRuntime)
    }
    
    private fun loadLabels() {
        labels = try {
            FileUtil.loadLabels(context, "labels.txt")
        } catch (e: Exception) {
            (0 until 1001).map { it.toString() }
        }
    }

    fun setModel(newModelName: String) {
        synchronized(lock) {
            if (modelName != newModelName) {
                modelName = newModelName
                loadModel(modelName, currentRuntime)
            }
        }
    }
    
    fun setRuntime(newRuntime: Runtime) {
        synchronized(lock) {
            if (currentRuntime != newRuntime) {
                currentRuntime = newRuntime
                modelWrapper.close()
                modelWrapper = when (newRuntime) {
                    Runtime.INTERPRETER -> InterpreterRunner(context)
                    Runtime.PLAY_SERVICES -> PlayServicesRunner(context)
                    Runtime.COMPILED_MODEL -> CompiledModelRunner(context)
                }
                loadModel(modelName, currentRuntime)
            }
        }
    }

    private fun loadModel(fileName: String, runtime: Runtime) {
        try {
            modelWrapper.load(fileName)
            
            val inputShape = modelWrapper.getInputShape()
            if (inputShape.size == 4) {
                if (ImageUtils.isNCHW(inputShape)) {
                    inputImageHeight = inputShape[2]
                    inputImageWidth = inputShape[3]
                } else {
                    inputImageHeight = inputShape[1]
                    inputImageWidth = inputShape[2]
                }
            }
            
            android.util.Log.d("ImageClassifier", "Loaded $fileName with $runtime. Input: ${modelWrapper.getInputDataType()}")

        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    fun classify(bitmap: Bitmap): InferenceResult {
        synchronized(lock) {
            // 인터프리터에서 직접 DataType 가져옴 (해킹 제거)
            val inputDataType = modelWrapper.getInputDataType()

            val imageProcessorBuilder = ImageProcessor.Builder()
                .add(ResizeOp(inputImageHeight, inputImageWidth, ResizeOp.ResizeMethod.BILINEAR))

            if (inputDataType == DataType.FLOAT32) {
                imageProcessorBuilder.add(NormalizeOp(127.5f, 127.5f))
            }

            val imageProcessor = imageProcessorBuilder.build()
            val tensorImage = TensorImage(inputDataType)
            tensorImage.load(bitmap)
            
            val (outputBuffer, time) = modelWrapper.classify(tensorImage, imageProcessor)
            
            // 출력 처리
            val outputArray = outputBuffer.floatArray
            val probabilities = softmax(outputArray)

            val topPredictions = probabilities
                .mapIndexed { index, value -> Pair(labels.getOrElse(index) { index.toString() }, value) }
                .sortedByDescending { it.second }
                .take(5)

            return InferenceResult(topPredictions, time)
        }
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.maxOrNull() ?: 0f
        val expValues = logits.map { exp((it - maxLogit).toDouble()).toFloat() }
        val sumExp = expValues.sum()
        return expValues.map { it / sumExp }.toFloatArray()
    }

    fun close() {
        synchronized(lock) {
            modelWrapper.close()
        }
    }
}
