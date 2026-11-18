// ==============================================================================
// ESP32 High-Performance ECG WebSocket Client for Clinical Telemetry
// ==============================================================================

#include <WiFi.h>
#include <WiFiMulti.h>
#include <WebSocketsClient.h>

// --- 1. MULTIPLE SAVED WI-FI NETWORKS ---
// Connects automatically to whichever network has the strongest signal!
struct WiFiCredential {
  const char* ssid;
  const char* pass;
};

const WiFiCredential KNOWN_NETWORKS[] = {
  {"JioFiber-DILIP", "air90000"},
  {"om", "12345678"},
  {"om", "7020344293"},
  {"MobileHotspot", "12345678"}
};
constexpr size_t NUM_NETWORKS = sizeof(KNOWN_NETWORKS) / sizeof(KNOWN_NETWORKS[0]);

// --- 2. VERIFIED RASPBERRY PI WEBSOCKET SERVER ---
const char* websocket_server_host = "192.168.29.237"; // Raspberry Pi 5 IP
const uint16_t websocket_server_port = 8080;          // Server Port

// --- 3. HARDWARE PINOUT & SAMPLING ---
const int ecgPin = 4; // AD8232 OUTPUT -> GPIO 4 (ADC1_CH3)
const int loPlusPin = 5; // AD8232 LO+ -> GPIO 5
const int loMinusPin = 6; // AD8232 LO- -> GPIO 6

constexpr uint32_t SAMPLE_INTERVAL_US = 4000; // 250 Hz (1,000,000 us / 250)
constexpr int BATCH_SIZE = 50;                // Send packet every 200ms for smooth live wave

// Set to 1 to enable USB Serial Plotter output (115200 baud)
#define ENABLE_SERIAL_PLOTTER 1

// --- GLOBALS ---
WiFiMulti wifiMulti;
WebSocketsClient webSocket;
int ecgBuffer[BATCH_SIZE];
int bufferIndex = 0;
uint32_t lastSampleMicros = 0;
float dcBaseline = 2048.0f;

void webSocketEvent(WStype_t type, uint8_t * payload, size_t length) {
  switch(type) {
    case WStype_DISCONNECTED:
      Serial.println("[WS] Disconnected from server!");
      break;
    case WStype_CONNECTED:
      Serial.println("[WS] >>> CONNECTED TO RASPBERRY PI TELEMETRY SERVER! <<<");
      break;
    case WStype_TEXT:
      // Feedback or commands from server
      break;
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n=== AD8232 High-Performance WebSocket Telemetry ===");

  // Configure Hardware Pins
  pinMode(ecgPin, INPUT);
  pinMode(loPlusPin, INPUT);
  pinMode(loMinusPin, INPUT);
  analogReadResolution(12);
  analogSetPinAttenuation(ecgPin, ADC_11db);

  // Disable Wi-Fi power-save to prevent RF noise on the analog 3.3V rail
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);

  for (size_t i = 0; i < NUM_NETWORKS; i++) {
    wifiMulti.addAP(KNOWN_NETWORKS[i].ssid, KNOWN_NETWORKS[i].pass);
  }

  Serial.println("[Wi-Fi] Scanning and connecting to saved networks...");
  uint32_t t0 = millis();
  while (wifiMulti.run() != WL_CONNECTED && millis() - t0 < 15000) {
    delay(250);
    Serial.print(".");
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("[Wi-Fi] Connected to '%s'! ESP32 IP: %s\n", 
                  WiFi.SSID().c_str(), WiFi.localIP().toString().c_str());
  } else {
    Serial.println("[Wi-Fi] Timeout — running in local Serial Plotter mode.");
  }

  // Connect to Raspberry Pi WebSocket Server
  Serial.printf("[WS] Connecting to ws://%s:%d/ws ...\n", websocket_server_host, websocket_server_port);
  webSocket.begin(websocket_server_host, websocket_server_port, "/ws");
  webSocket.onEvent(webSocketEvent);
  webSocket.setReconnectInterval(2000);

  lastSampleMicros = micros();
  Serial.println("[Init] Setup complete! Starting 250 Hz sampling...\n");
}

void sendECGBatch() {
  // Construct JSON batch: {"signal":[...]}
  String jsonStr;
  jsonStr.reserve(400);
  jsonStr = "{\"signal\":[";
  for (int i = 0; i < BATCH_SIZE; i++) {
    jsonStr += ecgBuffer[i];
    if (i < BATCH_SIZE - 1) {
      jsonStr += ",";
    }
  }
  jsonStr += "]}";

  if (webSocket.isConnected()) {
    webSocket.sendTXT(jsonStr);
  }
}

void loop() {
  webSocket.loop();

  // Maintain Wi-Fi connection
  if (WiFi.status() != WL_CONNECTED) {
    wifiMulti.run();
  }

  uint32_t now = micros();
  // Exact 250 Hz sample clock (every 4000 microseconds)
  if (now - lastSampleMicros >= SAMPLE_INTERVAL_US) {
    lastSampleMicros += SAMPLE_INTERVAL_US;

    int rawAdc = analogRead(ecgPin);

#if ENABLE_SERIAL_PLOTTER
    // Subtract DC baseline for local Serial Plotter display (-1.5 to +1.5 mV)
    dcBaseline = (0.995f * dcBaseline) + (0.005f * (float)rawAdc);
    float normalizedMv = ((float)rawAdc - dcBaseline) / 450.0f;
    Serial.println(normalizedMv, 4);
#endif

    // Store raw 12-bit ADC value into telemetry batch
    ecgBuffer[bufferIndex++] = rawAdc;

    // Send batch of 50 samples (every 200ms) to server
    if (bufferIndex >= BATCH_SIZE) {
      bufferIndex = 0;
      sendECGBatch();
    }
  }
}
