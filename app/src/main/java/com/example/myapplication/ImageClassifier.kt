package com.example.myapplication

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
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

    private val TAG = "ImageClassifier"
    private val lock = Any()

    private var interpreter: Interpreter? = null
    private var inputImageBuffer: TensorImage? = null
    private var outputProbabilityBuffer: TensorBuffer? = null
    
    private var cachedInputShape: IntArray = intArrayOf(1, 224, 224, 3)
    private var cachedOutputShape: IntArray = intArrayOf(1, 1001)
    private var inputDataType: DataType = DataType.FLOAT32
    private var outputDataType: DataType = DataType.FLOAT32
    private var outputQuantParams: org.tensorflow.lite.Tensor.QuantizationParams? = null
    
    private var inputScale: Float = 1.0f
    private var inputZeroPoint: Int = 0

    private var gpuDelegate: GpuDelegate? = null
    private var nnApiDelegate: NnApiDelegate? = null
    
    private var currentModelName: String = "mobilenet_v2.tflite"
    private var currentAccel: HardwareAccel = HardwareAccel.CPU
    private var isNCHW: Boolean = false
    private var inputBuffer: ByteBuffer? = null

    init {
        initialize()
    }

    fun initialize(modelName: String = "mobilenet_v2.tflite", accel: HardwareAccel = HardwareAccel.CPU) {
        synchronized(lock) {
            currentModelName = modelName
            currentAccel = accel
            loadModel()
        }
    }

    private fun loadModel() {
        Log.i(TAG, "Loading $currentModelName with $currentAccel")
        close()
        try {
            val modelBuffer = FileUtil.loadMappedFile(context, currentModelName)
            val options = Interpreter.Options()
            
            when (currentAccel) {
                HardwareAccel.GPU -> {
                    if (CompatibilityList().isDelegateSupportedOnThisDevice) {
                        gpuDelegate = GpuDelegate()
                        options.addDelegate(gpuDelegate)
                    }
                }
                HardwareAccel.NPU -> {
                    nnApiDelegate = NnApiDelegate()
                    options.addDelegate(nnApiDelegate)
                }
                HardwareAccel.CPU -> {
                    options.setUseXNNPACK(true)
                }
            }
            
            val newInterpreter = Interpreter(modelBuffer, options)
            val inputTensor = newInterpreter.getInputTensor(0)
            val outputTensor = newInterpreter.getOutputTensor(0)
            
            cachedInputShape = inputTensor.shape()
            cachedOutputShape = outputTensor.shape()
            inputDataType = inputTensor.dataType()
            outputDataType = outputTensor.dataType()
            
            if (inputDataType != DataType.FLOAT32) {
                val qp = inputTensor.quantizationParams()
                inputScale = qp.scale
                inputZeroPoint = qp.zeroPoint
                Log.d(TAG, "Input QuantParams: S=$inputScale, ZP=$inputZeroPoint")
            }

            if (outputDataType != DataType.FLOAT32) {
                try {
                    outputQuantParams = outputTensor.quantizationParams()
                    Log.d(TAG, "Output QuantParams: S=${outputQuantParams?.scale}, ZP=${outputQuantParams?.zeroPoint}")
                } catch (e: Exception) {
                    outputQuantParams = null
                }
            }
            
            isNCHW = cachedInputShape.size == 4 && cachedInputShape[1] == 3
            Log.i(TAG, "Model Ready: $currentModelName, isNCHW=$isNCHW, Input=$inputDataType")
            
            inputImageBuffer = TensorImage(inputDataType)
            outputProbabilityBuffer = TensorBuffer.createFixedSize(cachedOutputShape, outputDataType)
            
            val totalElements = cachedInputShape.reduce { acc, i -> acc * i }
            val bytesPerElement = if (inputDataType == DataType.FLOAT32) 4 else 1
            inputBuffer = ByteBuffer.allocateDirect(totalElements * bytesPerElement).order(ByteOrder.nativeOrder())
            
            interpreter = newInterpreter
            
        } catch (e: Exception) {
            Log.e(TAG, "Load failed: ${e.message}", e)
        }
    }

    fun classify(bitmap: Bitmap): Pair<String, Long> {
        synchronized(lock) {
            val currentInterpreter = interpreter ?: return "Error: Init Fail" to 0
            val startTime = android.os.SystemClock.uptimeMillis()

            try {
                val h = if (isNCHW) cachedInputShape[2] else cachedInputShape[1]
                val w = if (isNCHW) cachedInputShape[3] else cachedInputShape[2]
                
                val resizedBitmap = Bitmap.createScaledBitmap(bitmap, w, h, true)
                val pixels = IntArray(w * h)
                resizedBitmap.getPixels(pixels, 0, w, 0, 0, w, h)

                inputBuffer!!.rewind()
                val isF32 = inputDataType == DataType.FLOAT32
                val total = w * h

                if (isNCHW) {
                    if (isF32) {
                        val fb = inputBuffer!!.asFloatBuffer()
                        for (i in 0 until total) {
                            val p = pixels[i]
                            // R G B
                            fb.put(0 * total + i, (((p shr 16) and 0xFF) - 127.5f) / 127.5f)
                            fb.put(1 * total + i, (((p shr 8) and 0xFF) - 127.5f) / 127.5f)
                            fb.put(2 * total + i, ((p and 0xFF) - 127.5f) / 127.5f)
                        }
                    } else {
                        // UINT8 NCHW - Try B-G-R for INT8
                        for (i in 0 until total) {
                            val p = pixels[i]
                            val r = ((p shr 16) and 0xFF).toByte()
                            val g = ((p shr 8) and 0xFF).toByte()
                            val b = (p and 0xFF).toByte()
                            
                            // Swap R <-> B
                            inputBuffer!!.put(0 * total + i, b) // Blue first
                            inputBuffer!!.put(1 * total + i, g)
                            inputBuffer!!.put(2 * total + i, r) // Red last
                        }
                    }
                } else {
                    if (isF32) {
                        val fb = inputBuffer!!.asFloatBuffer()
                        for (p in pixels) {
                            fb.put((((p shr 16) and 0xFF) - 127.5f) / 127.5f)
                            fb.put((((p shr 8) and 0xFF) - 127.5f) / 127.5f)
                            fb.put(((p and 0xFF) - 127.5f) / 127.5f)
                        }
                    } else {
                        for (p in pixels) {
                            inputBuffer!!.put(((p shr 16) and 0xFF).toByte())
                            inputBuffer!!.put(((p shr 8) and 0xFF).toByte())
                            inputBuffer!!.put((p and 0xFF).toByte())
                        }
                    }
                }

                inputBuffer!!.rewind()
                currentInterpreter.run(inputBuffer, outputProbabilityBuffer!!.buffer.rewind())

                val labeledProb = getTopKLabels(outputProbabilityBuffer!!)
                val top1 = labeledProb.entries.firstOrNull()
                Log.i(TAG, "Prediction: ${top1?.key} (${(top1?.value ?: 0f) * 100}%)")
                
                val cost = android.os.SystemClock.uptimeMillis() - startTime
                
                val result = labeledProb.entries.joinToString("\n") { 
                    "${it.key} : ${(it.value * 100).toInt()}%" 
                }
                return result to cost

            } catch (e: Exception) {
                Log.e(TAG, "Classify fail: ${e.message}", e)
                return "Error: ${e.message}" to 0
            }
        }
    }

    private fun getTopKLabels(tensorBuffer: TensorBuffer): Map<String, Float> {
        val raw = tensorBuffer.floatArray
        
        // Diagnostic Log for INT8 saturation
        if (outputDataType != DataType.FLOAT32) {
             val rawInt = IntArray(raw.size) { raw[it].toInt() and 0xFF }
             val sorted = rawInt.mapIndexed { i, v -> i to v }.sortedByDescending { it.second }.take(5)
             Log.v(TAG, "INT8 raw distribution (top 5 indices/values): $sorted")
        }

        val dequantized = if (outputDataType == DataType.FLOAT32) {
            raw
        } else {
            val scale = outputQuantParams?.scale ?: 1.0f
            val zp = outputQuantParams?.zeroPoint ?: 0
            FloatArray(raw.size) { (raw[it] - zp) * scale }
        }

        val probs = softmax(dequantized)

        val labels = try {
            FileUtil.loadLabels(context, "labels.txt")
        } catch (e: Exception) {
            return mapOf("Error labels" to 0f)
        }

        return probs.mapIndexed { i, p -> i to p }
            .sortedByDescending { it.second }
            .take(3)
            .associate { (idx, p) ->
                val label = if (labels.size == 1000 && raw.size == 1001) {
                    if (idx == 0) "Background" else if (idx - 1 < labels.size) labels[idx - 1] else "Unknown($idx)"
                } else if (idx < labels.size) {
                    labels[idx]
                } else {
                    "Class $idx"
                }
                label to p
            }
    }

    private fun softmax(input: FloatArray): FloatArray {
        val max = input.maxOrNull() ?: 0.0f
        val exp = FloatArray(input.size) { kotlin.math.exp(input[it] - max).coerceAtLeast(0f) }
        val sum = exp.sum()
        return if (sum > 0.0f) FloatArray(input.size) { exp[it] / sum } else input
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

    fun setHardwareAccel(accel: HardwareAccel) = initialize(currentModelName, accel)
    fun setModel(modelName: String) = initialize(modelName, currentAccel)
}
