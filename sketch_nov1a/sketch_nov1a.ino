#include <WiFi.h>
#include <WiFiMulti.h>
#include <WiFiUdp.h>

// --- 1. MULTIPLE SAVED WI-FI NETWORKS ---
// Add as many networks as you like! The ESP32 will auto-connect to the best available network.
struct WiFiCredential {
  const char* ssid;
  const char* pass;
};

const WiFiCredential KNOWN_NETWORKS[] = {
  {"om", "12345678"},
  {"om", "7020344293"},
  {"MobileHotspot", "12345678"},
  {"HomeWiFi", "password"}
};
constexpr size_t NUM_NETWORKS = sizeof(KNOWN_NETWORKS) / sizeof(KNOWN_NETWORKS[0]);

const char* PI_IP     = "192.168.29.213"; // Raspberry Pi IP (auto-updated by auto_detect_pi.py)
const uint16_t UDP_PORT = 5005;

// --- 2. HARDWARE PINOUT (MATCHES PROTO-SHIELD SCHEMATIC) ---
constexpr int PIN_ECG_OUT = 4; // AD8232 OUTPUT -> GPIO 4 (ADC1_CH3)
constexpr int PIN_LO_PLUS = 5; // AD8232 LO+    -> GPIO 5
constexpr int PIN_LO_MINUS = 6; // AD8232 LO-    -> GPIO 6

// --- 3. 250 HZ TIMING & PACKET BUFFERING ---
constexpr uint32_t SAMPLE_INTERVAL_US = 4000; // 1,000,000 us / 250 Hz = 4000 us
constexpr int BATCH_SIZE = 25;                // Send packet every 100ms (25 samples)

WiFiMulti wifiMulti;
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

  for (size_t i = 0; i < NUM_NETWORKS; i++) {
    wifiMulti.addAP(KNOWN_NETWORKS[i].ssid, KNOWN_NETWORKS[i].pass);
  }

  Serial.println("Connecting to best available Wi-Fi...");
  uint32_t startAttempt = millis();
  while (wifiMulti.run() != WL_CONNECTED && millis() - startAttempt < 15000) {
    delay(250);
    Serial.print(".");
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("Connected to '%s'! ESP32 IP: %s\n", WiFi.SSID().c_str(), WiFi.localIP().toString().c_str());
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