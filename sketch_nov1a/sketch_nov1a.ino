#include <WiFi.h>
#include <WiFiUdp.h>

// --- 1. NETWORK CONFIGURATION ---
const char* WIFI_SSID = "om";
const char* WIFI_PASS = "12345678";
const char* PI_IP     = "10.20.211.48"; // Replace with your Raspberry Pi 5 IP
const uint16_t UDP_PORT = 5005;

// --- 2. HARDWARE PINOUT (MATCHES PROTO-SHIELD SCHEMATIC) ---
constexpr int PIN_ECG_OUT = 4; // AD8232 OUTPUT -> GPIO 4 (ADC1_CH3)
constexpr int PIN_LO_PLUS = 5; // AD8232 LO+    -> GPIO 5
constexpr int PIN_LO_MINUS = 6; // AD8232 LO-    -> GPIO 6

// --- 3. 250 HZ TIMING & PACKET BUFFERING ---
constexpr uint32_t SAMPLE_INTERVAL_US = 4000; // 1,000,000 us / 250 Hz = 4000 us
constexpr int BATCH_SIZE = 25;                // Send packet every 100ms (25 samples)

WiFiUDP udp;
uint32_t lastSampleMicros = 0;
float dcBaseline = 2048.0f; // 12-bit ADC midpoint (0-4095)
float sampleBatch[BATCH_SIZE];
int batchIndex = 0;

void setup() {
  Serial.begin(115200);

  // Configure Leads-Off Detection Pins
  pinMode(PIN_LO_PLUS, INPUT);
  pinMode(PIN_LO_MINUS, INPUT);

  // Configure ESP32-S3 12-Bit ADC (0V - 3.1V range)
  analogReadResolution(12);
  analogSetPinAttenuation(PIN_ECG_OUT, ADC_11db);

  // Disable Wi-Fi power-save to prevent 100ms beacon current spikes on the 3.3V analog rail
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  Serial.print("Connecting to Wi-Fi");
  uint32_t startAttempt = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - startAttempt < 10000) {
    delay(250);
    Serial.print(".");
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("Connected! ESP32-S3 IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("Wi-Fi timeout — running in local Serial Plotter mode.");
  }

  udp.begin(UDP_PORT);
  lastSampleMicros = micros();
}

void sendUdpBatch() {
  if (WiFi.status() != WL_CONNECTED) return;

  // Format as comma-separated floats matching server.py udp_listener()
  char payload[320];
  int offset = 0;
  for (int i = 0; i < BATCH_SIZE; i++) {
    int written = snprintf(
      payload + offset,
      sizeof(payload) - offset,
      (i < BATCH_SIZE - 1) ? "%.4f," : "%.4f",
      sampleBatch[i]
    );
    if (written > 0) offset += written;
  }

  udp.beginPacket(PI_IP, UDP_PORT);
  udp.write(reinterpret_cast<const uint8_t*>(payload), offset);
  udp.endPacket();
}

void loop() {
  uint32_t now = micros();

  // Strict 250 Hz non-blocking sample clock (4000 microseconds)
  if (now - lastSampleMicros >= SAMPLE_INTERVAL_US) {
    lastSampleMicros += SAMPLE_INTERVAL_US;

    float normalizedVal = 0.0f;

    // Check if any electrode pad is detached (LO+ or LO- goes HIGH)
    //if (digitalRead(PIN_LO_PLUS) == HIGH || digitalRead(PIN_LO_MINUS) == HIGH) {
    //  normalizedVal = 0.0f;
    //} else {
      int rawAdc = analogRead(PIN_ECG_OUT);

      // Slow exponential moving average to track and remove DC offset (~1.65V)
      dcBaseline = (0.995f * dcBaseline) + (0.005f * rawAdc);

      // Center around 0.0 and scale to match pipeline mV amplitude (~ -1.5 to +1.5)
      normalizedVal = (rawAdc - dcBaseline) / 450.0f;
   // }

    // 1. Output to USB Serial Plotter (115200 baud) for local bench verification
    Serial.println(normalizedVal, 4);

    // 2. Queue sample for UDP batch transmission
    sampleBatch[batchIndex++] = normalizedVal;
    if (batchIndex >= BATCH_SIZE) {
      batchIndex = 0;
      sendUdpBatch();
    }
  }
}