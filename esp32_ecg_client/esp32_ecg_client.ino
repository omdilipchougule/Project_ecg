// ESP32 ECG WebSocket Client

#include <WiFi.h>
#include <WebSocketsClient.h>

// WiFi credentials
const char* ssid = "om";
const char* password = "7020344293";

// WebSocket server details (Raspberry Pi IP)
const char* websocket_server_host = "192.168.39.48";
const uint16_t websocket_server_port = 8000;

// ECG configuration
const int ecgPin = 4; // ADC1 channel (GPIO 4)
const int sampleRate = 250; // Hz
const int bufferSize = 250; // 1 second buffer at 250Hz
constexpr uint32_t SAMPLE_INTERVAL_US = 4000; // 1,000,000 us / 250 Hz = 4000 us

// Global variables
WebSocketsClient webSocket;
int ecgBuffer[bufferSize];
int bufferIndex = 0;
uint32_t lastSampleMicros = 0;

void webSocketEvent(WStype_t type, uint8_t * payload, size_t length) {
  switch(type) {
    case WStype_DISCONNECTED:
      Serial.println("[WS] Disconnected!");
      break;
    case WStype_CONNECTED:
      Serial.println("[WS] Connected to Raspberry Pi server!");
      break;
    case WStype_TEXT:
      // Feedback from server if any
      break;
  }
}

void setup() {
  Serial.begin(115200);
  delay(500);

  // Configure ADC resolution & pin attenuation (0V - 3.1V range)
  analogReadResolution(12);
  analogSetPinAttenuation(ecgPin, ADC_11db);
  pinMode(ecgPin, INPUT);

  // Disable Wi-Fi power-save to prevent analog rail noise
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(ssid, password);

  Serial.print("Connecting to Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(250);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("Wi-Fi connected!");
  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());

  // Connect to Raspberry Pi WebSocket server
  webSocket.begin(websocket_server_host, websocket_server_port, "/ws");
  webSocket.onEvent(webSocketEvent);
  webSocket.setReconnectInterval(2000);

  lastSampleMicros = micros();
}

void loop() {
  webSocket.loop();

  uint32_t now = micros();
  // Non-blocking 250 Hz sample clock (every 4000 microseconds)
  if (now - lastSampleMicros >= SAMPLE_INTERVAL_US) {
    lastSampleMicros += SAMPLE_INTERVAL_US;

    // Read raw 12-bit ADC value (0 - 4095)
    ecgBuffer[bufferIndex++] = analogRead(ecgPin);

    // When 250 samples (1 full second) are collected, send to server
    if (bufferIndex >= bufferSize) {
      bufferIndex = 0;
      sendECGData();
    }
  }
}

void sendECGData() {
  // Construct JSON string directly: {"signal":[...]}
  String jsonStr;
  jsonStr.reserve(1500);
  jsonStr = "{\"signal\":[";
  for (int i = 0; i < bufferSize; i++) {
    jsonStr += ecgBuffer[i];
    if (i < bufferSize - 1) {
      jsonStr += ",";
    }
  }
  jsonStr += "]}";

  if (webSocket.isConnected()) {
    webSocket.sendTXT(jsonStr);
    Serial.println("[WS] 250-sample ECG batch transmitted");
  }
}
