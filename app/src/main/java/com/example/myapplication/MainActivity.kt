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

@Composable
fun MainScreen() {
    val context = LocalContext.current
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
            TabRow(selectedTabIndex = mode.ordinal) {
                Tab(selected = mode == AppMode.GALLERY, onClick = { mode = AppMode.GALLERY }) {
                    Text("Gallery", modifier = Modifier.padding(16.dp))
                }
                Tab(selected = mode == AppMode.CAMERA, onClick = { mode = AppMode.CAMERA }) {
                    Text("Camera", modifier = Modifier.padding(16.dp))
                }
            }
        }
    ) { innerPadding ->
        Box(modifier = Modifier.padding(innerPadding).fillMaxSize()) {
            if (hasCameraPermission) {
                when (mode) {
                    AppMode.GALLERY -> GalleryInferenceScreen()
                    AppMode.CAMERA -> CameraInferenceScreen()
                }
            } else {
                Text("Camera permission required", modifier = Modifier.align(Alignment.Center))
            }
        }
    }
}

enum class AppMode { GALLERY, CAMERA }

@Composable
fun GalleryInferenceScreen() {
    val context = LocalContext.current
    var selectedImageUri by remember { mutableStateOf<Uri?>(null) }
    var resultBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var results by remember { mutableStateOf<List<Pair<String, Float>>>(emptyList()) }
    var inferenceTime by remember { mutableStateOf(0L) }
    
    val classifier = remember { ImageClassifier(context) }

    val pickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent(),
        onResult = { uri ->
            selectedImageUri = uri
            uri?.let {
                val bitmap = MediaStore.Images.Media.getBitmap(context.contentResolver, it)
                resultBitmap = bitmap
                val res = classifier.classify(bitmap)
                results = res.predictions
                inferenceTime = res.inferenceTime
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

        ResultsList(results, inferenceTime)
    }
}

@Composable
fun CameraInferenceScreen() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    var results by remember { mutableStateOf<List<Pair<String, Float>>>(emptyList()) }
    var inferenceTime by remember { mutableStateOf(0L) }
    
    val classifier = remember { ImageClassifier(context) }
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
                                    results = res.predictions
                                    inferenceTime = res.inferenceTime
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
            ResultsList(results, inferenceTime)
        }
    }
}

@Composable
fun ResultsList(results: List<Pair<String, Float>>, inferenceTime: Long) {
    LazyColumn(modifier = Modifier.padding(16.dp)) {
        item {
            Text(
                text = "Inference Time: ${inferenceTime}ms",
                style = MaterialTheme.typography.labelLarge,
                modifier = Modifier.padding(bottom = 8.dp)
            )
        }
        items(results) { result ->
            Row(
                modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(text = result.first, style = MaterialTheme.typography.bodyLarge)
                Text(text = "%.2f%%".format(result.second * 100), style = MaterialTheme.typography.bodyLarge)
            }
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
