package com.ammarkov.dataacquisition
import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.ammarkov.dataacquisition.databinding.ActivityMainBinding
import com.ammarkov.dataacquisition.MyBackgroundService

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.startButton.setOnClickListener {
            startService()
        }

        binding.stopButton.setOnClickListener {
            stopService()
        }
    }

    private fun startService() {
        val serviceIntent = Intent(this, MyBackgroundService::class.java)
        startService(serviceIntent)
    }

    private fun stopService() {
        val serviceIntent = Intent(this, MyBackgroundService::class.java)
        stopService(serviceIntent)
    }
}