package com.example.myapplication

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.*
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.util.Size
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.example.myapplication.ui.theme.MyApplicationTheme
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {
    private lateinit var cameraExecutor: ExecutorService

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        cameraExecutor = Executors.newSingleThreadExecutor()
        setContent {
            MyApplicationTheme {
                MainScreen()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen() {
    val context = LocalContext.current
    var selectedModel by remember { mutableStateOf("mobilenet_v2.tflite") }
    // "Hardware Acceleration"
    var selectedAccel by remember { mutableStateOf(ImageClassifier.HardwareAccel.CPU) }
    
    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        )
    }
    val launcher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission(),
        onResult = { granted -> hasCameraPermission = granted }
    )

    LaunchedEffect(Unit) {
        if (!hasCameraPermission) {
            launcher.launch(Manifest.permission.CAMERA)
        }
    }

    var mode by remember { mutableStateOf(AppMode.GALLERY) }

    Scaffold(
        topBar = {
            Column {
                TabRow(selectedTabIndex = mode.ordinal) {
                    Tab(selected = mode == AppMode.GALLERY, onClick = { mode = AppMode.GALLERY }) {
                        Text("Gallery", modifier = Modifier.padding(16.dp))
                    }
                    Tab(selected = mode == AppMode.CAMERA, onClick = { mode = AppMode.CAMERA }) {
                        Text("Camera", modifier = Modifier.padding(16.dp))
                    }
                }
                // Model & Hardware Accel Selector
                Column(
                    modifier = Modifier.fillMaxWidth().padding(8.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    // Model Row
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.Center,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text("Model: ", style = MaterialTheme.typography.labelMedium)
                        RadioButton(
                            selected = selectedModel == "mobilenet_v2.tflite",
                            onClick = { selectedModel = "mobilenet_v2.tflite" }
                        )
                        Text("FP32", style = MaterialTheme.typography.bodySmall)
                        Spacer(modifier = Modifier.width(8.dp))
                        RadioButton(
                            selected = selectedModel == "mobilenet_v2_aiedge_int8.tflite",
                            onClick = { selectedModel = "mobilenet_v2_aiedge_int8.tflite" }
                        )
                        Text("INT8", style = MaterialTheme.typography.bodySmall)
                    }
                    
                    // Hardware Acceleration Row
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.Center,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text("Accel: ", style = MaterialTheme.typography.labelMedium)
                        
                        RadioButton(
                            selected = selectedAccel == ImageClassifier.HardwareAccel.CPU,
                            onClick = { selectedAccel = ImageClassifier.HardwareAccel.CPU }
                        )
                        Text("CPU", style = MaterialTheme.typography.bodySmall)
                        Spacer(modifier = Modifier.width(8.dp))
                        
                        RadioButton(
                            selected = selectedAccel == ImageClassifier.HardwareAccel.GPU,
                            onClick = { selectedAccel = ImageClassifier.HardwareAccel.GPU }
                        )
                        Text("GPU", style = MaterialTheme.typography.bodySmall)
                        Spacer(modifier = Modifier.width(8.dp))
                        
                        RadioButton(
                            selected = selectedAccel == ImageClassifier.HardwareAccel.NPU,
                            onClick = { selectedAccel = ImageClassifier.HardwareAccel.NPU }
                        )
                        Text("NPU", style = MaterialTheme.typography.bodySmall)
                    }
                }
            }
        }
    ) { innerPadding ->
        Box(modifier = Modifier.padding(innerPadding).fillMaxSize()) {
            if (hasCameraPermission) {
                when (mode) {
                    AppMode.GALLERY -> GalleryInferenceScreen(selectedModel, selectedAccel)
                    AppMode.CAMERA -> CameraInferenceScreen(selectedModel, selectedAccel)
                }
            } else {
                Text("Camera permission required", modifier = Modifier.align(Alignment.Center))
            }
        }
    }
}

enum class AppMode { GALLERY, CAMERA }

@Composable
fun GalleryInferenceScreen(modelName: String, accel: ImageClassifier.HardwareAccel) {
    val context = LocalContext.current
    var selectedImageUri by remember { mutableStateOf<Uri?>(null) }
    var resultBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var resultString by remember { mutableStateOf("") }
    var inferenceTime by remember { mutableStateOf(0L) }
    
    val classifier = remember { ImageClassifier(context) }
    
    // Update model when selection changes (Model OR Accel)
    LaunchedEffect(modelName) {
        classifier.setModel(modelName)
    }
    LaunchedEffect(accel) {
        classifier.setHardwareAccel(accel)
    }

    val pickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent(),
        onResult = { uri ->
            selectedImageUri = uri
            uri?.let {
                val bitmap = MediaStore.Images.Media.getBitmap(context.contentResolver, it)
                resultBitmap = bitmap
                val res = classifier.classify(bitmap)
                resultString = res.first
                inferenceTime = res.second
            }
        }
    )

    Column(
        modifier = Modifier.fillMaxSize().padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Button(onClick = { pickerLauncher.launch("image/*") }) {
            Text("Select Image from Gallery")
        }
        
        Spacer(modifier = Modifier.height(16.dp))

        resultBitmap?.let {
            Image(
                bitmap = it.asImageBitmap(),
                contentDescription = null,
                modifier = Modifier.height(300.dp).fillMaxWidth()
            )
        }

        Spacer(modifier = Modifier.height(16.dp))

        ResultsList(resultString, inferenceTime)
    }
}

@Composable
fun CameraInferenceScreen(modelName: String, accel: ImageClassifier.HardwareAccel) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    var resultString by remember { mutableStateOf("") }
    var inferenceTime by remember { mutableStateOf(0L) }
    var averageTime by remember { mutableStateOf(0L) }
    
    // Thread-safe history buffer: (Timestamp, Duration)
    val inferenceHistory = remember { java.util.Collections.synchronizedList(java.util.ArrayList<Pair<Long, Long>>()) }
    
    val classifier = remember { ImageClassifier(context) }
    
    // Update model when selection changes
    LaunchedEffect(modelName) {
        classifier.setModel(modelName)
        inferenceHistory.clear()
        averageTime = 0L
    }
    LaunchedEffect(accel) {
        classifier.setHardwareAccel(accel)
        inferenceHistory.clear()
        averageTime = 0L
    }

    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

    Box(modifier = Modifier.fillMaxSize()) {
        AndroidView(
            factory = { ctx ->
                val previewView = PreviewView(ctx)
                val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)

                cameraProviderFuture.addListener({
                    val cameraProvider = cameraProviderFuture.get()
                    val preview = Preview.Builder().build().also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }

                    val imageAnalyzer = ImageAnalysis.Builder()
                        .setTargetResolution(Size(224, 224))
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build()
                        .also {
                            it.setAnalyzer(cameraExecutor) { imageProxy ->
                                val bitmap = imageProxy.toBitmap()
                                if (bitmap != null) {
                                    val res = classifier.classify(bitmap)
                                    val currentDuration = res.second
                                    resultString = res.first
                                    inferenceTime = currentDuration
                                    
                                    // 10s Moving Average Logic
                                    val now = System.currentTimeMillis()
                                    synchronized(inferenceHistory) {
                                        inferenceHistory.add(Pair(now, currentDuration))
                                        // Remove entries older than 10 seconds (10000 ms)
                                        val cutoff = now - 10000
                                        inferenceHistory.removeIf { it.first < cutoff }
                                        
                                        val sum = inferenceHistory.sumOf { it.second }
                                        averageTime = if (inferenceHistory.isNotEmpty()) sum / inferenceHistory.size else 0L
                                    }
                                }
                                imageProxy.close()
                            }
                        }

                    val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

                    try {
                        cameraProvider.unbindAll()
                        cameraProvider.bindToLifecycle(
                            lifecycleOwner, cameraSelector, preview, imageAnalyzer
                        )
                    } catch (e: Exception) {
                        Log.e("CameraInference", "Use case binding failed", e)
                    }
                }, ContextCompat.getMainExecutor(ctx))
                previewView
            },
            modifier = Modifier.fillMaxSize()
        )

        Surface(
            color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.8f),
            modifier = Modifier.align(Alignment.BottomCenter).fillMaxWidth().padding(16.dp)
        ) {
            ResultsList(resultString, inferenceTime, averageTime)
        }
    }
}

@Composable
fun ResultsList(resultString: String, inferenceTime: Long, averageTime: Long = 0L) {
    LazyColumn(modifier = Modifier.padding(16.dp)) {
        item {
            Row(modifier = Modifier.fillMaxWidth().padding(bottom = 8.dp), horizontalArrangement = Arrangement.SpaceBetween) {
                Text(
                    text = "Time: ${inferenceTime}ms",
                    style = MaterialTheme.typography.labelLarge,
                )
                if (averageTime > 0) {
                    Text(
                        text = "Avg(10s): ${averageTime}ms",
                        style = MaterialTheme.typography.labelLarge,
                        color = MaterialTheme.colorScheme.primary
                    )
                }
            }
        }
        item {
            Text(
                text = resultString,
                style = MaterialTheme.typography.bodyLarge
            )
        }
    }
}

fun ImageProxy.toBitmap(): Bitmap? {
    val plane = planes[0]
    val buffer = plane.buffer
    val bytes = ByteArray(buffer.remaining())
    buffer.get(bytes)
    
    // This is a simplified conversion. For production, use YUV to RGB conversion if needed.
    // However, with setTargetResolution(224, 224), we might get JPEG or need manual conversion.
    // CameraX ImageAnalysis usually provides YUV_420_888.
    
    // Improved YUV to Bitmap conversion helper
    val yBuffer = planes[0].buffer
    val uBuffer = planes[1].buffer
    val vBuffer = planes[2].buffer

    val ySize = yBuffer.remaining()
    val uSize = uBuffer.remaining()
    val vSize = vBuffer.remaining()

    val nv21 = ByteArray(ySize + uSize + vSize)

    yBuffer.get(nv21, 0, ySize)
    vBuffer.get(nv21, ySize, vSize)
    uBuffer.get(nv21, ySize + vSize, uSize)

    val yuvImage = YuvImage(nv21, ImageFormat.NV21, this.width, this.height, null)
    val out = java.io.ByteArrayOutputStream()
    yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 100, out)
    val imageBytes = out.toByteArray()
    val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    
    // Rotate bitmap if necessary
    val matrix = Matrix()
    matrix.postRotate(imageInfo.rotationDegrees.toFloat())
    return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
}
