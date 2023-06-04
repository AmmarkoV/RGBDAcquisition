package com.ammarkov.dataacquisition

import android.content.Context
import android.os.Environment
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.*

object DataLogger {
    private const val JSON_FILE_NAME = "data.json"
    private val dateFormat = SimpleDateFormat("yyyy-MM-dd", Locale.getDefault())

    fun logData(context: Context, data: JSONObject) {
        val jsonArray = readDataArray(context)
        jsonArray.put(data)
        writeDataArray(context, jsonArray)
    }

    fun readDataArray(context: Context): JSONArray {
        val file = getDataFile(context)
        if (!file.exists()) {
            file.createNewFile()
            return JSONArray()
        }
        val json = file.readText()
        return JSONArray(json)
    }

    private fun writeDataArray(context: Context, jsonArray: JSONArray) {
        val file = getDataFile(context)
        val writer = FileWriter(file)
        writer.use {
            writer.write(jsonArray.toString())
        }
    }

    private fun getDataFile(context: Context): File {
        val rootDir = getRootDirectory()
        val dateDir = getTodayDirectory(rootDir)
        return File(dateDir, JSON_FILE_NAME)
    }

    private fun getRootDirectory(): File {
        val rootDir = Environment.getExternalStorageDirectory()
        val dataAcquisitionDir = File(rootDir, "DataAcquisition")
        dataAcquisitionDir.mkdirs()
        return dataAcquisitionDir
    }

    private fun getTodayDirectory(rootDir: File): File {
        val today = dateFormat.format(Date())
        val todayDir = File(rootDir, today)
        todayDir.mkdirs()
        return todayDir
    }

    fun createDataEntry(
        location: String,
        velocity: Float,
        altitude: Double,
        accelerometerData: JSONObject,
        imageFilePath: String?
    ): JSONObject {
        val dataEntry = JSONObject()
        dataEntry.put("location", location)
        dataEntry.put("velocity", velocity)
        dataEntry.put("altitude", altitude)
        dataEntry.put("accelerometerData", accelerometerData)
        if (imageFilePath != null) {
            dataEntry.put("imageFilePath", imageFilePath)
        }
        return dataEntry
    }
}
