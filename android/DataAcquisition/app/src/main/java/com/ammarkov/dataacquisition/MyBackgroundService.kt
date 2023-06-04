package com.ammarkov.dataacquisition

import android.Manifest
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.hardware.Camera
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Bundle
import android.os.Handler
import android.os.IBinder
import android.util.Log
import androidx.core.app.ActivityCompat
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.*

class MyBackgroundService : Service() {

    private lateinit var locationManager: LocationManager
    private lateinit var sensorManager: SensorManager
    private lateinit var accelerometerSensor: Sensor
    private lateinit var camera: Camera

    private val locationListener: LocationListener = object : LocationListener {
        override fun onLocationChanged(location: Location) {
            val dataEntry = DataLogger.createDataEntry(
                "${location.latitude},${location.longitude}",
                location.speed,
                location.altitude,
                getAccelerometerData(),
                captureImage()
            )
            DataLogger.logData(applicationContext, dataEntry)
        }

        override fun onStatusChanged(provider: String, status: Int, extras: Bundle) {}
        override fun onProviderEnabled(provider: String) {}
        override fun onProviderDisabled(provider: String) {}
    }

    private val accelerometerListener: SensorEventListener = object : SensorEventListener {
        override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) {}

        override fun onSensorChanged(event: SensorEvent) {
            if (event.sensor.type == Sensor.TYPE_ACCELEROMETER) {
                // Process accelerometer data and store it in a JSON object
            }
        }
    }

    private lateinit var handler: Handler // Declare handler as a class-level variable

    override fun onBind(intent: Intent): IBinder? {
        return null
    }

    override fun onCreate() {
        super.onCreate()
        locationManager = getSystemService(Context.LOCATION_SERVICE) as LocationManager
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        accelerometerSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        camera = openCamera()
        handler = Handler() // Initialize the handler
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (ActivityCompat.checkSelfPermission(
                this,
                Manifest.permission.ACCESS_FINE_LOCATION
            ) != PackageManager.PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(
                this,
                Manifest.permission.ACCESS_COARSE_LOCATION
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            return START_NOT_STICKY
        }
        locationManager.requestLocationUpdates(
            LocationManager.GPS_PROVIDER,
            0,
            0f,
            locationListener
        )
        sensorManager.registerListener(
            accelerometerListener,
            accelerometerSensor,
            SensorManager.SENSOR_DELAY_NORMAL
        )
        startImageCapture()
        return START_STICKY
    }

    override fun onDestroy() {
        super.onDestroy()
        locationManager.removeUpdates(locationListener)
        sensorManager.unregisterListener(accelerometerListener)
        stopImageCapture()
        releaseCamera()
    }

    private fun openCamera(): Camera {
        val cameraId = Camera.CameraInfo.CAMERA_FACING_BACK
        return Camera.open(cameraId)
    }

    private fun releaseCamera() {
        camera.release()
    }

    private fun startImageCapture() {
        val captureIntervalMillis = 5000L // Set the capture interval here (5000 milliseconds in this example)
        handler.postDelayed(object : Runnable {
            override fun run() {
                captureImage()
                handler.postDelayed(this, captureIntervalMillis)
            }
        }, captureIntervalMillis)
    }

    private fun stopImageCapture() {
        // Remove any pending capture tasks from the handler
        handler.removeCallbacksAndMessages(null)
    }

    private fun captureImage(): String? {
        try {
            camera.startPreview()
            val imageFile = saveImageToFile()
            return imageFile?.absolutePath
        } catch (e: Exception) {
            Log.e(TAG, "Error capturing image: ${e.message}")
        }
        return null
    }

    private fun saveImageToFile(): File? {
        val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
        val fileName = "IMG_$timeStamp.jpg"
        val storageDir = getExternalFilesDir(null)
        val imageFile = File(storageDir, fileName)

        try {
            camera.takePicture(null, null) { data, _ ->
                try {
                    val outputStream = FileOutputStream(imageFile)
                    outputStream.write(data)
                    outputStream.close()
                } catch (e: Exception) {
                    Log.e(TAG, "Error saving image: ${e.message}")
                }
            }
            return imageFile
        } catch (e: Exception) {
            Log.e(TAG, "Error capturing image: ${e.message}")
        }
        return null
    }

    private fun getAccelerometerData(): JSONObject {
        // Retrieve and process accelerometer data and return it as a JSON object
        return JSONObject()
    }

    companion object {
        private const val TAG = "MyBackgroundService"
    }
}
