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

class ImageClassifier(context: Context) {

    private var interpreter: Interpreter? = null
    private var labels: List<String> = emptyList()

    private val inputImageWidth: Int
    private val inputImageHeight: Int
    private val modelName = "mobilenet_v2.tflite" // Ensure this matches your file name in assets

    init {
        val model: MappedByteBuffer = FileUtil.loadMappedFile(context, modelName)
        val options = Interpreter.Options()
        interpreter = Interpreter(model, options)

        val inputShape = interpreter?.getInputTensor(0)?.shape() // Can be [1, H, W, 3] or [1, 3, H, W]
        
        if (inputShape != null && inputShape.size == 4) {
            if (inputShape[1] == 3) {
                // NCHW format
                inputImageHeight = inputShape[2]
                inputImageWidth = inputShape[3]
            } else {
                // NHWC format
                inputImageHeight = inputShape[1]
                inputImageWidth = inputShape[2]
            }
        } else {
             inputImageHeight = 224
             inputImageWidth = 224
        }

        labels = try {
            FileUtil.loadLabels(context, "labels.txt")
        } catch (e: Exception) {
            (0 until 1001).map { it.toString() }
        }
    }

    fun classify(bitmap: Bitmap): InferenceResult {
        var inferenceTime = 0L
        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(inputImageHeight, inputImageWidth, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(127.5f, 127.5f))
            .build()

        var tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmap)
        tensorImage = imageProcessor.process(tensorImage)

        val outputShape = interpreter?.getOutputTensor(0)?.shape()
        val probabilityBuffer = TensorBuffer.createFixedSize(outputShape, DataType.FLOAT32)

        if (inputImageHeight == 224 && inputImageWidth == 224 && interpreter?.getInputTensor(0)?.shape()?.get(1) == 3) {
            // Detected NCHW model [1, 3, 224, 224], but TensorImage is NHWC [1, 224, 224, 3]
            // We must convert NHWC -> NCHW
            val floatBuffer = tensorImage.buffer.asFloatBuffer()
            val nhwcArray = FloatArray(floatBuffer.remaining())
            floatBuffer.get(nhwcArray)

            val nchwArray = FloatArray(nhwcArray.size)
            val height = 224
            val width = 224
            
            // Permute logic (same as before)
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
            val startTime = android.os.SystemClock.uptimeMillis()
            interpreter?.run(tensorImage.buffer, probabilityBuffer.buffer)
            val endTime = android.os.SystemClock.uptimeMillis()
            inferenceTime = endTime - startTime
        }

        // Apply Softmax to probabilityBuffer (which contains logits)
        val logits = probabilityBuffer.floatArray
        val probabilities = softmax(logits)

        // Create a map manually since TensorLabel expects buffer. 
        // We can just map indices to labels manually using our probabilities.
        val labeledProbabilities = labels.zip(probabilities.toList())
        
        val sortedResults = labeledProbabilities.sortedByDescending { it.second }.take(3)
        return InferenceResult(sortedResults, inferenceTime)
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
        interpreter?.close()
        interpreter = null
    }
}

data class InferenceResult(
    val predictions: List<Pair<String, Float>>,
    val inferenceTime: Long
)
