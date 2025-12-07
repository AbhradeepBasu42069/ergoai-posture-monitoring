// Posture & Spine Health Monitor
// ESP32 + MPU6050 + vibration motor + buzzer
// Auto-calibrates at startup (sit straight while it says "Calibrating")

#include <Wire.h>
#include <MPU6050_tockn.h>

MPU6050 mpu(Wire);

// ---------- Optional WiFi / HTTP POST settings ----------
// To enable posting telemetry to your Flask server, set ENABLE_WIFI_POST to 1
// and configure SSID/PASS and SERVER_URL below. Keep disabled if using USB Serial only.
#define ENABLE_WIFI_POST 1

#if ENABLE_WIFI_POST
#include <WiFi.h>
#include <HTTPClient.h>
const char* WIFI_SSID = "realme 9i";
const char* WIFI_PASS = "t2rt2fm9";
const char* SERVER_URL = "http://10.79.110.224:5000/api/hardware_update"; // PC local IP
unsigned long lastPostMs = 0;
const unsigned long POST_INTERVAL_MS = 500; // post every 500 ms
#endif

// === Pin assignments ===
const int SDA_PIN      = 21;
const int SCL_PIN      = 22;
const int VIB_PIN      = 25;  // Motor driver (BJT base via 1k)
const int BUZZER_PIN   = 26;  // Active buzzer (direct or via driver)

// === Sampling & posture parameters ===
const float SAMPLE_HZ        = 100.0f;          // 100 samples per second
const unsigned long DT_MS    = 1000 / SAMPLE_HZ;

const int   CAL_SAMPLES      = 500;             // ~5 seconds at 100 Hz
const float WARN_ANGLE_DEG   = 2.0f;            // soft warning threshold
const float ALERT_ANGLE_DEG  = 4.0f;           // strong slouch alert
const float HYSTERESIS_DEG   = 2.0f;            // to prevent flicker
const int   HOLD_MS          = 1500;            // must slouch this long to alert

// Moving average filter window for smoother angle (0.3–0.5 s)
const int   SMOOTH_WINDOW    = 40;              // 40 samples @ 100 Hz = 0.4 s

// === Globals ===
float baselinePitch = 0.0f;
bool  calibrated    = false;

// Circular buffer for smoothing
float smoothBuf[SMOOTH_WINDOW];
int   smoothIndex = 0;
int   smoothCount = 0;

// State for timing & alert logic
unsigned long lastSampleMs = 0;
unsigned long slouchStartMs = 0;
bool alertActive = false;

// -------- Helper: moving average filter --------
float smooth(float x) {
  smoothBuf[smoothIndex] = x;
  smoothIndex = (smoothIndex + 1) % SMOOTH_WINDOW;
  if (smoothCount < SMOOTH_WINDOW) smoothCount++;

  float sum = 0.0f;
  for (int i = 0; i < smoothCount; i++) {
    sum += smoothBuf[i];
  }
  return sum / smoothCount;
}

// -------- Calibration: user sits straight for a few seconds --------
void calibrateBaseline() {
  Serial.println("\n=== Calibration ===");
  Serial.println("Sit/stand straight and do NOT move for ~5 seconds...");
  delay(1000);

  float sumPitch = 0.0f;

  for (int i = 0; i < CAL_SAMPLES; i++) {
    mpu.update();

    // mpu.getAngleX() is usually the pitch (depends on lib orientation)
    // If you find it inverted in tests, multiply by -1.
    float pitch = mpu.getAngleX();

    sumPitch += pitch;

    delay(DT_MS);
  }

  baselinePitch = sumPitch / CAL_SAMPLES;
  calibrated = true;

  // Serial.print("Baseline pitch = ");
  // Serial.print(baselinePitch);
  // Serial.println(" deg");
  // Serial.println("Calibration done. Start moving normally.\n");
}

// -------- Alert helpers --------
void motorOn() {
  digitalWrite(VIB_PIN, HIGH);
}

void motorOff() {
  digitalWrite(VIB_PIN, LOW);
}

void buzzerBeep(int ms) {
  digitalWrite(BUZZER_PIN, HIGH);
  delay(ms);
  digitalWrite(BUZZER_PIN, LOW);
}

// -------- Arduino setup --------
void setup() {
  Serial.begin(115200);
  delay(1000);

#if ENABLE_WIFI_POST
  Serial.println("WiFi POST enabled — attempting to connect...");
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 10000) {
    Serial.print('.');
    delay(500);
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println();
    Serial.print("WiFi connected. Local IP: ");
    Serial.println(WiFi.localIP());
    Serial.print("Posting telemetry to: ");
    Serial.println(SERVER_URL);
  } else {
    Serial.println();
    Serial.println("WiFi connect failed — falling back to Serial-only output.");
  }
#else
  Serial.println("WiFi POST disabled. Telemetry will be printed to Serial only.");
#endif

  // I2C + IMU init
  Wire.begin(SDA_PIN, SCL_PIN);
  mpu.begin();
  // This library does internal calibration, keep the board still
  // Serial.println("MPU6050 init, calculating gyro offsets..."); // Keep output clean
  mpu.calcGyroOffsets(true);
  delay(500);

  // Pins
  pinMode(VIB_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  motorOff();
  digitalWrite(BUZZER_PIN, LOW);

  // Start with empty smoothing buffer
  for (int i = 0; i < SMOOTH_WINDOW; i++) smoothBuf[i] = 0.0f;

  // Auto calibration on startup
  calibrateBaseline();

  lastSampleMs = millis();
}

// -------- Arduino loop --------
void loop() {
  unsigned long now = millis();
  if (now - lastSampleMs < DT_MS) {
    return;  // wait until next sample time
  }
  lastSampleMs = now;

  // Read IMU
  mpu.update();

  // Using getAngleX() as pitch; adjust if needed in testing
  float pitch = mpu.getAngleX(); // degrees
  float slouchRaw = pitch - baselinePitch;   // +ve when leaning forward (depending on orientation)
  float slouch = smooth(slouchRaw);          // smoothed angle

  // --- Debug output for Serial Plotter (original Arduino IDE output) ---
  Serial.print(now);
  Serial.print(", pitch=");
  Serial.print(pitch, 2);
  Serial.print(", slouch=");
  Serial.print(slouch, 2);
  Serial.print(", alert=");
  Serial.println(alertActive ? 1 : 0);

  // --- HTTP POST to Flask server (if enabled) ---
#if ENABLE_WIFI_POST
  if (WiFi.status() == WL_CONNECTED && millis() - lastPostMs >= POST_INTERVAL_MS) {
    lastPostMs = millis();
    
      // Build JSON payload for server
      char jsonPayload[256];
      snprintf(jsonPayload, sizeof(jsonPayload), 
           "{\"pitch\":%.2f,\"slouch\":%.2f,\"alert\":%d,\"connected\":true}",
           pitch, slouch, alertActive ? 1 : 0);
    
    HTTPClient http;
    http.begin(SERVER_URL);
    http.addHeader("Content-Type", "application/json");
    int httpCode = http.POST((uint8_t*)jsonPayload, strlen(jsonPayload));

    if (httpCode == 200) {
      // Success - silent to keep Serial output clean for Arduino IDE monitoring
    } else {
      Serial.print("Server POST error: ");
      Serial.println(httpCode);
    }
    http.end();
  }
#endif

  // --- Posture logic ---

  // Soft warning region
  if (!alertActive && slouch > WARN_ANGLE_DEG && slouch < ALERT_ANGLE_DEG) {
    // Optional: short buzz or quick motor tick
    // motorOn(); delay(50); motorOff();
  }

  // Hard alert region (slouch beyond ALERT_ANGLE_DEG)
  if (!alertActive) {
    if (slouch > ALERT_ANGLE_DEG) {
      if (slouchStartMs == 0) {
        slouchStartMs = now;   // start timing
      } else if (now - slouchStartMs >= HOLD_MS) {
        // Condition held long enough -> trigger alert
        alertActive = true;
        motorOn();
        buzzerBeep(150); // short beep
      }
    } else {
      // Not slouching enough -> reset timer
      slouchStartMs = 0;
    }
  } else {
    // While in alert state: clear only when slouch < (ALERT_ANGLE_DEG - HYSTERESIS_DEG)
    if (slouch < (ALERT_ANGLE_DEG - HYSTERESIS_DEG)) {
      alertActive = false;
      slouchStartMs = 0;
      motorOff();
      // Serial.println(">>> Posture improved, alert cleared."); // Commented out to keep JSON clean
    }
  }
}

  // (Optional) Energy saving: you could put ESP32 into light sleep if needed

