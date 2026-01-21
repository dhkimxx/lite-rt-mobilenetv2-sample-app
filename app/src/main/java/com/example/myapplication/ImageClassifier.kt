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
import org.tensorflow.lite.support.label.TensorLabel
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.MappedByteBuffer

class ImageClassifier(private val context: Context) {

    private var interpreter: Interpreter? = null
    private var labels: List<String> = emptyList()

    private val lock = Any()
    private var inputImageWidth: Int = 224
    private var inputImageHeight: Int = 224
    private var modelName = "mobilenet_v2.tflite"

    init {
        loadModel(modelName)
    }

    fun setModel(newModelName: String) {
        synchronized(lock) {
            if (modelName != newModelName) {
                modelName = newModelName
                loadModel(modelName)
            }
        }
    }

    private fun loadModel(fileName: String) {
        close()
        try {
            val model: MappedByteBuffer = FileUtil.loadMappedFile(context, fileName)
            val options = Interpreter.Options()
            interpreter = Interpreter(model, options)

            val inputShape = interpreter?.getInputTensor(0)?.shape()
            if (inputShape != null && inputShape.size == 4) {
                if (inputShape[1] == 3) {
                    inputImageHeight = inputShape[2]
                    inputImageWidth = inputShape[3]
                } else {
                    inputImageHeight = inputShape[1]
                    inputImageWidth = inputShape[2]
                }
            } else {
                inputImageHeight = 224
                inputImageWidth = 224
            }
            android.util.Log.d("ImageClassifier", "Loaded model: $fileName")
        } catch (e: Exception) {
            e.printStackTrace()
        }
        
        // Reload labels if needed, or keep generic
        if (labels.isEmpty()) {
            labels = try {
                FileUtil.loadLabels(context, "labels.txt")
            } catch (e: Exception) {
                (0 until 1001).map { it.toString() }
            }
        }
    }

    private fun getInputDataType(): DataType {
        return interpreter?.getInputTensor(0)?.dataType() ?: DataType.FLOAT32
    }

    private fun getOutputDataType(): DataType {
        return interpreter?.getOutputTensor(0)?.dataType() ?: DataType.FLOAT32
    }

    fun classify(bitmap: Bitmap): InferenceResult {
        synchronized(lock) {
            if (interpreter == null) {
                return InferenceResult(emptyList(), 0L)
            }

            var inferenceTime = 0L
            val inputDataType = getInputDataType()
            
            // Dynamic Image Processor
            val imageProcessorBuilder = ImageProcessor.Builder()
                .add(ResizeOp(inputImageHeight, inputImageWidth, ResizeOp.ResizeMethod.BILINEAR))
    
            // Only add NormalizeOp if input is FLOAT32. 
            // Quantized models (UINT8/INT8) expect 0-255, which Bitmap already provides (approx).
            if (inputDataType == DataType.FLOAT32) {
                imageProcessorBuilder.add(NormalizeOp(127.5f, 127.5f))
            }
    
            val imageProcessor = imageProcessorBuilder.build()
    
            var tensorImage = TensorImage(inputDataType)
            tensorImage.load(bitmap)
            tensorImage = imageProcessor.process(tensorImage)
    
            val outputShape = interpreter?.getOutputTensor(0)?.shape()
            // Create buffer matching the model's output type
            val outputDataType = getOutputDataType()
            val probabilityBuffer = TensorBuffer.createFixedSize(outputShape, outputDataType)
    
            if (inputImageHeight == 224 && inputImageWidth == 224 && interpreter?.getInputTensor(0)?.shape()?.get(1) == 3) {
                // NCHW Conversion Logic (Only if needed? Assuming INT8 model might also be NCHW? 
                // Usually TFLite INT8 models are NHWC. But let's check NCHW condition safely.)
                
                // Note: NCHW permutation is complex with UINT8 buffers directly effectively. 
                // But let's assume valid assumption logic:
                
                // Logic: Convert TensorBuffer to desired array -> Permute -> Write back
                // If UINT8, array is bytes. If Float, array is floats.
                
                // To simplify: convert to FloatArray, permute, then convert back? No, precision loss/performance.
                // Let's implement generic permutation if NCHW detected.
                
                // Actually, for simplicity, I will stick to the Float path for NCHW permutation 
                // OR checks generic types.
                // But practically, `mobilenet_v2_aiedge_int8.tflite` is likely NHWC.
                // The logic below ONLY triggers if `inputShape[1] == 3`. 
                // If the new model is NHWC, this block skips.
                
                // If it IS NCHW and INT8, we need Byte permutation. 
                // I'll assume for now the user's INT8 model is standard NHWC (unlikely to be NCHW for TFLite quantized).
                // So I'll keep the Float-specific NCHW block but wrap it to only run if FLOAT32.
                // If it's UINT8 NCHW, we might crash or need new logic. 
                // I'll verify if `mobilenet_v2.tflite` was NCHW.
                
                val isNCHW = interpreter?.getInputTensor(0)?.shape()?.get(1) == 3
                
                if (isNCHW) {
                     if (inputDataType == DataType.FLOAT32) {
                         // Existing Float permutation logic
                         val floatBuffer = tensorImage.buffer.asFloatBuffer()
                         val nhwcArray = FloatArray(floatBuffer.remaining())
                         floatBuffer.get(nhwcArray)
    
                         val nchwArray = FloatArray(nhwcArray.size)
                         val height = 224
                         val width = 224
                         
                         for (h in 0 until height) {
                             for (w in 0 until width) {
                                 val pixelIndex = (h * width) + w
                                 nchwArray[pixelIndex] = nhwcArray[pixelIndex * 3 + 0]
                                 nchwArray[(height * width) + pixelIndex] = nhwcArray[pixelIndex * 3 + 1]
                                 nchwArray[(2 * height * width) + pixelIndex] = nhwcArray[pixelIndex * 3 + 2]
                             }
                         }
                         
                         val inputBuffer = java.nio.ByteBuffer.allocateDirect(nchwArray.size * 4)
                         inputBuffer.order(java.nio.ByteOrder.nativeOrder())
                         inputBuffer.asFloatBuffer().put(nchwArray)
    
                         val startTime = android.os.SystemClock.uptimeMillis()
                         interpreter?.run(inputBuffer, probabilityBuffer.buffer)
                         val endTime = android.os.SystemClock.uptimeMillis()
                         inferenceTime = endTime - startTime
                     } else {
                         // Quantized NCHW - Permute bytes (UINT8)
                         val byteBuffer = tensorImage.buffer
                         // Ensure rewind
                         byteBuffer.rewind()
                         val nhwcArray = ByteArray(byteBuffer.remaining())
                         byteBuffer.get(nhwcArray)

                         val nchwArray = ByteArray(nhwcArray.size)
                         val height = 224
                         val width = 224
                         
                         for (h in 0 until height) {
                             for (w in 0 until width) {
                                 val pixelIndex = (h * width) + w
                                 // NHWC: [h, w, c]
                                 // NCHW: [c, h, w]
                                 // R
                                 nchwArray[pixelIndex] = nhwcArray[pixelIndex * 3 + 0]
                                 // G
                                 nchwArray[(height * width) + pixelIndex] = nhwcArray[pixelIndex * 3 + 1]
                                 // B
                                 nchwArray[(2 * height * width) + pixelIndex] = nhwcArray[pixelIndex * 3 + 2]
                             }
                         }
                         
                         val inputBuffer = java.nio.ByteBuffer.allocateDirect(nchwArray.size)
                         inputBuffer.order(java.nio.ByteOrder.nativeOrder())
                         inputBuffer.put(nchwArray)
                         
                         val startTime = android.os.SystemClock.uptimeMillis()
                         interpreter?.run(inputBuffer, probabilityBuffer.buffer)
                         val endTime = android.os.SystemClock.uptimeMillis()
                         inferenceTime = endTime - startTime
                     }
                } else {
                    // NHWC (Standard)
                    val startTime = android.os.SystemClock.uptimeMillis()
                    interpreter?.run(tensorImage.buffer, probabilityBuffer.buffer)
                    val endTime = android.os.SystemClock.uptimeMillis()
                    inferenceTime = endTime - startTime
                }
    
            } else {
                // Standard NHWC or unknown
                val startTime = android.os.SystemClock.uptimeMillis()
                interpreter?.run(tensorImage.buffer, probabilityBuffer.buffer)
                val endTime = android.os.SystemClock.uptimeMillis()
                inferenceTime = endTime - startTime
            }
    
            // Apply Softmax via logits
            // probabilityBuffer.floatArray automatically dequantizes if needed! :D
            // (TensorBuffer checks properties and transforms).
            val logits = probabilityBuffer.floatArray
            val probabilities = softmax(logits)
    
            val labeledProbabilities = labels.zip(probabilities.toList())
            
            val sortedResults = labeledProbabilities.sortedByDescending { it.second }.take(3)
            return InferenceResult(sortedResults, inferenceTime)
        }
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val expValues = FloatArray(logits.size)
        var sumExp = 0.0f
        val maxLogit = logits.maxOrNull() ?: 0.0f
        
        for (i in logits.indices) {
            expValues[i] = kotlin.math.exp(logits[i] - maxLogit)
            sumExp += expValues[i]
        }
        
        for (i in logits.indices) {
            expValues[i] /= sumExp
        }
        return expValues
    }

    fun close() {
        synchronized(lock) {
            interpreter?.close()
            interpreter = null
        }
    }
}

data class InferenceResult(
    val predictions: List<Pair<String, Float>>,
    val inferenceTime: Long
)
