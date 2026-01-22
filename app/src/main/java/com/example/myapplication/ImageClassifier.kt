package com.example.myapplication

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder

class ImageClassifier(private val context: Context) {

    enum class HardwareAccel {
        CPU, GPU, NPU
    }

    private val lock = Any()

    init {
        initialize()
    }

    private var interpreter: Interpreter? = null
    private var inputImageBuffer: TensorImage? = null
    private var outputProbabilityBuffer: TensorBuffer? = null
    
    // 캐시된 모델 정보
    private var cachedInputShape: IntArray = intArrayOf(1, 224, 224, 3)
    private var cachedOutputShape: IntArray = intArrayOf(1, 1001)
    private var inputDataType: DataType = DataType.FLOAT32
    private var outputDataType: DataType = DataType.FLOAT32
    
    // Delegate 참조 (닫기 위해 필요)
    private var gpuDelegate: GpuDelegate? = null
    private var nnApiDelegate: NnApiDelegate? = null
    
    // 현재 설정
    private var currentModelName: String = "mobilenet_v2.tflite"
    private var currentAccel: HardwareAccel = HardwareAccel.CPU

    fun initialize(modelName: String = "mobilenet_v2.tflite", accel: HardwareAccel = HardwareAccel.CPU) {
        synchronized(lock) {
            currentModelName = modelName
            currentAccel = accel
            loadModel()
        }
    }

    private var isNCHW: Boolean = false
    private var inputBuffer: ByteBuffer? = null

    private fun loadModel() {
        close() // Internal close, safe because initialize holds lock
        try {
            val modelBuffer = FileUtil.loadMappedFile(context, currentModelName)
            val options = Interpreter.Options()
            
            when (currentAccel) {
                HardwareAccel.GPU -> {
                    if (CompatibilityList().isDelegateSupportedOnThisDevice) {
                        gpuDelegate = GpuDelegate()
                        options.addDelegate(gpuDelegate)
                        android.util.Log.d("ImageClassifier", "Initialized GPU Delegate")
                    } else {
                        android.util.Log.w("ImageClassifier", "GPU Delegate not supported on this device. Falling back to CPU.")
                    }
                }
                HardwareAccel.NPU -> {
                    nnApiDelegate = NnApiDelegate()
                    options.addDelegate(nnApiDelegate)
                    android.util.Log.d("ImageClassifier", "Initialized NNApi Delegate")
                }
                HardwareAccel.CPU -> {
                    options.setUseXNNPACK(true) // XNNPACK은 CPU 가속을 위해 기본적으로 활성화 권장
                    android.util.Log.d("ImageClassifier", "Initialized CPU (XNNPACK)")
                }
            }
            interpreter = Interpreter(modelBuffer, options)
            
            // 텐서 정보 읽기
            val inputTensor = interpreter!!.getInputTensor(0)
            val outputTensor = interpreter!!.getOutputTensor(0)
            
            cachedInputShape = inputTensor.shape()
            cachedOutputShape = outputTensor.shape()
            inputDataType = inputTensor.dataType()
            outputDataType = outputTensor.dataType()
            
            // NCHW 감지 ( [1, 3, H, W] )
            isNCHW = cachedInputShape.size == 4 && cachedInputShape[1] == 3
            
            if (isNCHW && currentAccel != HardwareAccel.CPU) {
                // Warning: NCHW on GPU/NPU might be unstable.
                android.util.Log.w("ImageClassifier", "NCHW model detected with GPU/NPU. If crash occurs, please switch to CPU.")
            }

            android.util.Log.d("ImageClassifier", "Model loaded: $currentModelName, Input: $inputDataType, Shape: ${cachedInputShape.contentToString()}, NCHW: $isNCHW")
            
            // 버퍼 초기화
            // TensorImage는 NHWC를 다루지만, NCHW 변환을 위한 별도 ByteBuffer가 필요할 수 있음
            inputImageBuffer = TensorImage(inputDataType)
            outputProbabilityBuffer = TensorBuffer.createFixedSize(cachedOutputShape, outputDataType)
            
            if (isNCHW) {
                val inputSize = cachedInputShape.reduce { acc, i -> acc * i }
                val bytesPerElement = if (inputDataType == DataType.FLOAT32) 4 else 1
                inputBuffer = ByteBuffer.allocateDirect(inputSize * bytesPerElement).order(ByteOrder.nativeOrder())
            }
            
        } catch (e: Exception) {
            android.util.Log.e("ImageClassifier", "Error loading model: ${e.message}")
            e.printStackTrace()
        }
    }

    fun classify(bitmap: Bitmap): Pair<String, Long> {
        synchronized(lock) {
            if (interpreter == null) return Pair("Error: Model not loaded", 0)

            val startTime = android.os.SystemClock.uptimeMillis()

            // 1. 이미지 전처리 (입력 Shape에 맞게 Resize)
            val height = if (isNCHW) cachedInputShape[2] else cachedInputShape[1]
            val width = if (isNCHW) cachedInputShape[3] else cachedInputShape[2]

            val imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(height, width, ResizeOp.ResizeMethod.BILINEAR))
                .add(NormalizeOp(127.5f, 127.5f)) // [0, 255] -> [-1, 1]
                .build()

            inputImageBuffer!!.load(bitmap)
            val processedImage = imageProcessor.process(inputImageBuffer)

            // 2. 추론
            try {
                if (isNCHW) {
                    // NHWC -> NCHW 변환
                    if (inputBuffer == null) throw IllegalStateException("Input buffer not initialized for NCHW")
                    inputBuffer!!.rewind()

                    if (inputDataType == DataType.FLOAT32) {
                        ImageUtils.permuteNHWCToNCHW_Float(processedImage.buffer, height, width, inputBuffer!!)
                    } else {
                        ImageUtils.permuteNHWCToNCHW_Uint8(processedImage.buffer, height, width, inputBuffer!!)
                    }

                    inputBuffer!!.rewind()
                    interpreter!!.run(inputBuffer, outputProbabilityBuffer!!.buffer.rewind())
                } else {
                    // NHWC 그대로 전달
                    interpreter!!.run(processedImage.buffer, outputProbabilityBuffer!!.buffer.rewind())
                }
            } catch (e: Exception) {
                android.util.Log.e("ImageClassifier", "Inference failed: ${e.message}")
                return Pair("Error: Inference failed", 0)
            }

            val inferenceTime = android.os.SystemClock.uptimeMillis() - startTime

            // 3. 결과 후처리
            val labeledProbability = getTopKLabels(outputProbabilityBuffer!!)

            val resultString = labeledProbability.entries.joinToString(separator = "\n") { 
                "${it.key} : ${(it.value * 100).toInt()}%" 
            }

            return Pair(resultString, inferenceTime)
        }
    }

    object ImageUtils {
        // NHWC(Bitmap) -> NCHW(Model Input) 변환 (Float)
        fun permuteNHWCToNCHW_Float(src: ByteBuffer, height: Int, width: Int, dst: ByteBuffer) {
            // src: [H, W, 3]
            // dst: [3, H, W]
            val totalPixels = height * width
            src.rewind()
            val srcFloat = src.asFloatBuffer()
            val dstFloat = dst.asFloatBuffer()
            
            val rBase = 0
            val gBase = totalPixels
            val bBase = totalPixels * 2
            
            for (i in 0 until totalPixels) {
                val r = srcFloat.get(i * 3 + 0)
                val g = srcFloat.get(i * 3 + 1)
                val b = srcFloat.get(i * 3 + 2)
                
                dstFloat.put(rBase + i, r)
                dstFloat.put(gBase + i, g)
                dstFloat.put(bBase + i, b)
            }
        }
        
        // NHWC(Bitmap) -> NCHW(Model Input) 변환 (Uint8)
        fun permuteNHWCToNCHW_Uint8(src: ByteBuffer, height: Int, width: Int, dst: ByteBuffer) {
            val totalPixels = height * width
            src.rewind()
            val rBase = 0
            val gBase = totalPixels
            val bBase = totalPixels * 2
            
            for (i in 0 until totalPixels) {
                val r = src.get(i * 3 + 0)
                val g = src.get(i * 3 + 1)
                val b = src.get(i * 3 + 2)
                
                dst.put(rBase + i, r)
                dst.put(gBase + i, g)
                dst.put(bBase + i, b)
            }
        }
    }
    
    private fun getTopKLabels(tensorBuffer: TensorBuffer): Map<String, Float> {
        val probabilities = tensorBuffer.floatArray
        // 확률 분포로 변환 (필요 시 Softmax)
        val softmaxProb = softmax(probabilities)
        
        val maxIndex = softmaxProb.indices.maxByOrNull { softmaxProb[it] } ?: -1
        if (maxIndex == -1) return emptyMap()
        
        // 라벨 로드 (실제 앱에서는 파일에서 한 번만 읽도록 최적화 권장)
        val labels = try {
            FileUtil.loadLabels(context, "labels.txt")
        } catch (e: Exception) {
            return mapOf("Unknown" to softmaxProb[maxIndex])
        }
        
        val topK = softmaxProb.mapIndexed { index, fl -> index to fl }
            .sortedByDescending { it.second }
            .take(3)
            
        val result = mutableMapOf<String, Float>()
        val outputSize = probabilities.size
        
        for ((index, prob) in topK) {
            var label = "Unknown($index)"
            if (labels.size == 1000 && outputSize == 1001) {
                // ImageNet 1001 class model with 1000 labels (Index 0 is background)
                if (index > 0 && index - 1 < labels.size) {
                    label = labels[index - 1]
                } else if (index == 0) {
                    label = "Background"
                }
            } else {
                // Standard mapping
                if (index < labels.size) {
                    label = labels[index]
                }
            }
            result[label] = prob
        }
        return result
    }
    
    // Softmax 함수 구현
    private fun softmax(input: FloatArray): FloatArray {
        val output = FloatArray(input.size)
        var sum = 0.0f
        val max = input.maxOrNull() ?: 0.0f
        
        for (i in input.indices) {
            output[i] = kotlin.math.exp(input[i] - max)
            sum += output[i]
        }
        
        if (sum != 0.0f) {
            for (i in output.indices) {
                output[i] /= sum
            }
        }
        return output
    }

    fun close() {
        synchronized(lock) {
            interpreter?.close()
            interpreter = null
            gpuDelegate?.close()
            gpuDelegate = null
            nnApiDelegate?.close()
            nnApiDelegate = null
        }
    }
    
    fun setHardwareAccel(accel: HardwareAccel) {
        if (currentAccel != accel) {
            initialize(currentModelName, accel)
        }
    }
    
    fun setModel(modelName: String) {
        if (currentModelName != modelName) {
            initialize(modelName, currentAccel)
        }
    }
}
