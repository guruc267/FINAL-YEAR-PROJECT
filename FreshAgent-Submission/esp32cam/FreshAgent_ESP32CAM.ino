/*
 * ═══════════════════════════════════════════════════════════════════════════
 * FreshAgent – ESP32-CAM Module
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * PURPOSE:
 *   Captures JPEG images using the OV2640 camera on the ESP32-CAM (AI-Thinker)
 *   and POSTs them as multipart/form-data to the FreshAgent FastAPI backend
 *   at the endpoint:  POST /api/esp32-stream
 *
 *   The ESP32-CAM does NO AI inference. It only sends raw JPEG bytes.
 *   All classification (Fresh / Rotten / Formalin / Adulterated) happens
 *   on the server side using the ONNX model.
 *
 * BOARD SETUP (Arduino IDE):
 *   1. File → Preferences → Additional Board Manager URLs:
 *      https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
 *   2. Tools → Board → ESP32 Arduino → AI Thinker ESP32-CAM
 *   3. Tools → Partition Scheme → Huge APP (3MB No OTA / 1MB SPIFFS)
 *   4. Tools → Upload Speed → 115200
 *   5. Use an FTDI adapter (USB-to-Serial) to upload.
 *      Connect: FTDI TX → ESP32 U0R,  FTDI RX → ESP32 U0T
 *      Short GPIO0 to GND before uploading, then remove after upload.
 *
 * CONFIGURATION:
 *   - Set your WiFi SSID and password below.
 *   - Set the server IP address (the PC running the FreshAgent backend).
 *   - Adjust CAPTURE_INTERVAL_MS for how often images are sent.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>

// ═══════════════════════════════════════════════════════════════════════════
// ██  USER CONFIGURATION — CHANGE THESE  ██
// ═══════════════════════════════════════════════════════════════════════════

// WiFi credentials
const char* WIFI_SSID     = "tata";              // ← Your WiFi network name
const char* WIFI_PASSWORD = "srm@123";           // ← Your WiFi password

// FreshAgent server address (the PC running main.py)
// Find your PC's IP: open CMD and type "ipconfig", look for IPv4 Address
const char* SERVER_HOST = "192.168.1.24";           // ← Your PC's local IP
const int   SERVER_PORT = 9090;                      // ← Must match main.py port

// How often to capture and send an image (in milliseconds)
// 3000 = every 3 seconds.  Lower = faster but more load on server & network.
const unsigned long CAPTURE_INTERVAL_MS = 3000;

// ═══════════════════════════════════════════════════════════════════════════
// ██  AI-THINKER ESP32-CAM PIN DEFINITIONS  ██
// ═══════════════════════════════════════════════════════════════════════════
// DO NOT CHANGE these unless you have a different ESP32-CAM board.

#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5

#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// On-board LED (flash) on GPIO 4
#define FLASH_LED_PIN      4

// ═══════════════════════════════════════════════════════════════════════════
// ██  GLOBAL VARIABLES  ██
// ═══════════════════════════════════════════════════════════════════════════

unsigned long lastCaptureTime = 0;
bool cameraReady = false;

// ═══════════════════════════════════════════════════════════════════════════
// ██  CAMERA INITIALIZATION  ██
// ═══════════════════════════════════════════════════════════════════════════

bool initCamera() {
  // ── Step 1: Explicitly power-cycle the camera via PWDN pin ──────────
  // The PWDN (power down) pin on AI-Thinker is GPIO 32.
  // Setting it HIGH powers DOWN the camera, LOW powers it UP.
  // This hard-reset ensures the OV2640 sensor is in a known state.
  Serial.println("[CAM] Power-cycling camera module...");
  pinMode(PWDN_GPIO_NUM, OUTPUT);
  digitalWrite(PWDN_GPIO_NUM, HIGH);  // Power DOWN camera
  delay(300);                          // Hold power down for 300ms
  digitalWrite(PWDN_GPIO_NUM, LOW);   // Power UP camera
  delay(300);                          // Wait for sensor to stabilize

  camera_config_t config;

  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;

  config.xclk_freq_hz = 10000000;       // 10 MHz XCLK — lower for stability
  config.pixel_format = PIXFORMAT_JPEG;  // JPEG output — required for HTTP POST
  config.grab_mode    = CAMERA_GRAB_LATEST;  // Always grab the newest frame

  // Use higher resolution if PSRAM is available, otherwise fall back
  if (psramFound()) {
    Serial.println("[CAM] PSRAM detected — using high quality settings");
    config.frame_size   = FRAMESIZE_VGA;   // 640×480 — good balance of quality vs speed
    config.jpeg_quality = 12;              // 0-63, lower = better quality, more bytes
    config.fb_count     = 2;               // Double buffer for smoother capture
  } else {
    Serial.println("[CAM] No PSRAM — using lower resolution");
    config.frame_size   = FRAMESIZE_CIF;   // 400×296
    config.jpeg_quality = 16;
    config.fb_count     = 1;
  }

  // ── Step 2: Try camera init up to 3 times with increasing delays ────
  esp_err_t err = ESP_FAIL;
  for (int attempt = 1; attempt <= 3; attempt++) {
    Serial.printf("[CAM] Initialization attempt %d of 3...\n", attempt);
    err = esp_camera_init(&config);
    if (err == ESP_OK) {
      break;  // Success!
    }
    Serial.printf("[CAM] Attempt %d failed with error 0x%x\n", attempt, err);
    
    if (attempt < 3) {
      // Power-cycle again before retry
      esp_camera_deinit();
      digitalWrite(PWDN_GPIO_NUM, HIGH);
      delay(500 * attempt);  // Increasing delay: 500ms, 1000ms
      digitalWrite(PWDN_GPIO_NUM, LOW);
      delay(500 * attempt);
      Serial.println("[CAM] Retrying after power-cycle...");
    }
  }

  if (err != ESP_OK) {
    Serial.printf("[CAM] ERROR: Camera init failed after 3 attempts (last error 0x%x)\n", err);
    Serial.println("[CAM] >>> Possible causes:");
    Serial.println("[CAM]   1. Camera ribbon cable is loose — reseat it firmly");
    Serial.println("[CAM]   2. Camera module is defective — try a different OV2640");
    Serial.println("[CAM]   3. Insufficient power — use 5V/2A supply, NOT 3.3V");
    Serial.println("[CAM]   4. Wrong board — must be 'AI Thinker ESP32-CAM' in Arduino IDE");
    return false;
  }

  // Fine-tune camera sensor settings for food photography
  sensor_t *s = esp_camera_sensor_get();
  if (s != NULL) {
    s->set_brightness(s, 1);     // Slightly brighter
    s->set_contrast(s, 1);       // Slightly more contrast
    s->set_saturation(s, 1);     // Slightly more saturated colors
    s->set_whitebal(s, 1);       // Enable auto white balance
    s->set_awb_gain(s, 1);       // Enable AWB gain
    s->set_wb_mode(s, 0);        // Auto WB mode
    s->set_exposure_ctrl(s, 1);  // Enable auto exposure
    s->set_aec2(s, 1);           // Enable AEC DSP
    s->set_gain_ctrl(s, 1);      // Enable auto gain
    s->set_agc_gain(s, 0);       // AGC gain = 0 (auto)
    s->set_gainceiling(s, (gainceiling_t)6);  // Max gain ceiling
    s->set_bpc(s, 1);            // Enable black pixel correction
    s->set_wpc(s, 1);            // Enable white pixel correction
    s->set_raw_gma(s, 1);        // Enable gamma correction
    s->set_lenc(s, 1);           // Enable lens correction
    s->set_dcw(s, 1);            // Enable downsize EN
  }

  Serial.println("[CAM] Camera initialized successfully");
  return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// ██  WIFI CONNECTION  ██
// ═══════════════════════════════════════════════════════════════════════════

void connectWiFi() {
  Serial.printf("[WiFi] Connecting to '%s'", WIFI_SSID);

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    attempts++;

    if (attempts > 40) {  // 20 seconds timeout
      Serial.println("\n[WiFi] ERROR: Connection failed after 20 seconds!");
      Serial.println("[WiFi] Please check SSID and password, then reset the board.");
      // Blink LED rapidly to indicate error
      for (int i = 0; i < 10; i++) {
        digitalWrite(FLASH_LED_PIN, HIGH);
        delay(100);
        digitalWrite(FLASH_LED_PIN, LOW);
        delay(100);
      }
      ESP.restart();
    }
  }

  Serial.println();
  Serial.println("[WiFi] ✓ Connected!");
  Serial.printf("[WiFi]   IP Address : %s\n", WiFi.localIP().toString().c_str());
  Serial.printf("[WiFi]   Signal (RSSI): %d dBm\n", WiFi.RSSI());

  // Brief LED flash to confirm connection
  digitalWrite(FLASH_LED_PIN, HIGH);
  delay(300);
  digitalWrite(FLASH_LED_PIN, LOW);
}

// ═══════════════════════════════════════════════════════════════════════════
// ██  SEND IMAGE TO SERVER  ██
// ═══════════════════════════════════════════════════════════════════════════

bool sendImageToServer(camera_fb_t *fb) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[HTTP] WiFi not connected — reconnecting...");
    connectWiFi();
  }

  // Build the server URL
  String serverUrl = "http://";
  serverUrl += SERVER_HOST;
  serverUrl += ":";
  serverUrl += String(SERVER_PORT);
  serverUrl += "/api/esp32-stream";

  HTTPClient http;
  http.begin(serverUrl);
  http.setTimeout(10000);  // 10 second timeout

  // ─── Build multipart/form-data body ─────────────────────────────────────
  String boundary = "----FreshAgentBoundary";
  String contentType = "multipart/form-data; boundary=" + boundary;
  http.addHeader("Content-Type", contentType);

  // Construct the multipart body
  // Part 1: Header before the binary data
  String bodyStart = "--" + boundary + "\r\n";
  bodyStart += "Content-Disposition: form-data; name=\"file\"; filename=\"esp32cam.jpg\"\r\n";
  bodyStart += "Content-Type: image/jpeg\r\n\r\n";

  // Part 2: Footer after the binary data
  String bodyEnd = "\r\n--" + boundary + "--\r\n";

  // Calculate total content length
  size_t totalLen = bodyStart.length() + fb->len + bodyEnd.length();

  // Allocate buffer for the complete body
  uint8_t *payload = (uint8_t *)malloc(totalLen);
  if (payload == NULL) {
    Serial.println("[HTTP] ERROR: Failed to allocate memory for HTTP payload");
    http.end();
    return false;
  }

  // Assemble the payload
  size_t offset = 0;
  memcpy(payload + offset, bodyStart.c_str(), bodyStart.length());
  offset += bodyStart.length();
  memcpy(payload + offset, fb->buf, fb->len);
  offset += fb->len;
  memcpy(payload + offset, bodyEnd.c_str(), bodyEnd.length());

  // ─── Send the POST request ─────────────────────────────────────────────
  Serial.printf("[HTTP] Sending %u bytes to %s ... ", totalLen, serverUrl.c_str());

  int httpCode = http.POST(payload, totalLen);
  free(payload);  // Free the buffer immediately

  if (httpCode > 0) {
    Serial.printf("Response: %d\n", httpCode);

    if (httpCode == 200) {
      String response = http.getString();
      // Print a truncated version of the response (avoid flooding serial)
      if (response.length() > 200) {
        Serial.println("[HTTP] ✓ Server response (truncated): " + response.substring(0, 200) + "...");
      } else {
        Serial.println("[HTTP] ✓ Server response: " + response);
      }
      http.end();
      return true;
    } else {
      Serial.printf("[HTTP] Server returned error code: %d\n", httpCode);
      String errorBody = http.getString();
      Serial.println("[HTTP] Error body: " + errorBody);
    }
  } else {
    Serial.printf("[HTTP] POST failed, error: %s\n", http.errorToString(httpCode).c_str());
    Serial.println("[HTTP] Make sure the FreshAgent server is running at:");
    Serial.printf("[HTTP]   %s:%d\n", SERVER_HOST, SERVER_PORT);
  }

  http.end();
  return false;
}

// ═══════════════════════════════════════════════════════════════════════════
// ██  CAPTURE AND SEND  ██
// ═══════════════════════════════════════════════════════════════════════════

void captureAndSend() {
  // Capture a frame from the camera
  camera_fb_t *fb = esp_camera_fb_get();
  if (fb == NULL) {
    Serial.println("[CAM] ERROR: Frame capture failed");
    return;
  }

  Serial.printf("[CAM] Captured frame: %u bytes (%dx%d)\n", fb->len, fb->width, fb->height);

  // Send to server
  bool success = sendImageToServer(fb);

  // Return the frame buffer so it can be reused
  esp_camera_fb_return(fb);

  if (success) {
    // Quick LED blink on successful send
    digitalWrite(FLASH_LED_PIN, HIGH);
    delay(50);
    digitalWrite(FLASH_LED_PIN, LOW);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// ██  SETUP  ██
// ═══════════════════════════════════════════════════════════════════════════

void setup() {
  Serial.begin(115200);
  Serial.println();
  Serial.println("═══════════════════════════════════════════════════");
  Serial.println("  FreshAgent – ESP32-CAM Module v1.0");
  Serial.println("  Fruit & Vegetable Adulteration Detection");
  Serial.println("═══════════════════════════════════════════════════");
  Serial.println();

  // Initialize flash LED pin
  pinMode(FLASH_LED_PIN, OUTPUT);
  digitalWrite(FLASH_LED_PIN, LOW);

  // Initialize camera
  Serial.println("[INIT] Step 1/2: Initializing camera...");
  cameraReady = initCamera();
  if (!cameraReady) {
    Serial.println("[INIT] FATAL: Camera initialization failed!");
    Serial.println("[INIT] Check wiring and board selection (AI Thinker ESP32-CAM).");
    // Blink SOS pattern
    while (true) {
      for (int i = 0; i < 3; i++) { digitalWrite(FLASH_LED_PIN, HIGH); delay(200); digitalWrite(FLASH_LED_PIN, LOW); delay(200); }
      delay(400);
      for (int i = 0; i < 3; i++) { digitalWrite(FLASH_LED_PIN, HIGH); delay(600); digitalWrite(FLASH_LED_PIN, LOW); delay(200); }
      delay(400);
      for (int i = 0; i < 3; i++) { digitalWrite(FLASH_LED_PIN, HIGH); delay(200); digitalWrite(FLASH_LED_PIN, LOW); delay(200); }
      delay(2000);
    }
  }

  // Connect to WiFi
  Serial.println("[INIT] Step 2/2: Connecting to WiFi...");
  connectWiFi();

  // Ready!
  Serial.println();
  Serial.println("═══════════════════════════════════════════════════");
  Serial.println("  ✓ FreshAgent ESP32-CAM Ready!");
  Serial.printf("  Camera    : OK\n");
  Serial.printf("  WiFi      : %s (%d dBm)\n", WiFi.localIP().toString().c_str(), WiFi.RSSI());
  Serial.printf("  Server    : http://%s:%d/api/esp32-stream\n", SERVER_HOST, SERVER_PORT);
  Serial.printf("  Interval  : %lu ms\n", CAPTURE_INTERVAL_MS);
  Serial.println("═══════════════════════════════════════════════════");
  Serial.println();
  Serial.println("[LOOP] Starting image capture loop...");

  // Take one throwaway frame — first frame is often dark/garbled
  camera_fb_t *fb = esp_camera_fb_get();
  if (fb) {
    esp_camera_fb_return(fb);
    Serial.println("[CAM] Discarded warm-up frame");
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// ██  MAIN LOOP  ██
// ═══════════════════════════════════════════════════════════════════════════

void loop() {
  unsigned long now = millis();

  // Check if it's time to capture
  if (now - lastCaptureTime >= CAPTURE_INTERVAL_MS) {
    lastCaptureTime = now;
    captureAndSend();
  }

  // Check WiFi health periodically
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[WiFi] Connection lost — reconnecting...");
    connectWiFi();
  }

  delay(10);  // Small delay to avoid watchdog issues
}
