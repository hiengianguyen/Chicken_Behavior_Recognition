#include "DHT.h"
#include "ArduinoJson.h"

#define DHTPIN A1        // Chân DATA của DHT11
#define DHTTYPE DHT11   // Loại cảm biến

#define RELAY_PIN_QUAT 3     // Chân điều khiển relay
#define RELAY_PIN_DEN 5     // Chân điều khiển relay
#define RELAY_PIN_HEATER 6
#define RELAY_PIN_MIST 7
#define MOTOR_PIN 8     // Chân điều khiển đồ ăn tự động
#define WINDOW_STEP_PIN 9
#define WINDOW_DIR_PIN 10
#define WINDOW_STEPS_PER_MOVE 200 // Calibrate to the window's full travel.

const int gasPin = A0; // Chân analog nối với module gas
const int threshold = 400; // Mức giới hạn cảnh báo (tùy chỉnh)

int sensorValue = 0;      // Biến lưu giá trị đọc được

DHT dht(DHTPIN, DHTTYPE);

void setup() {
  Serial.begin(9600);
  dht.begin();

  pinMode(RELAY_PIN_QUAT, OUTPUT);
  pinMode(RELAY_PIN_DEN, OUTPUT);
  pinMode(RELAY_PIN_HEATER, OUTPUT);
  pinMode(RELAY_PIN_MIST, OUTPUT);
  pinMode(MOTOR_PIN, OUTPUT);
  pinMode(WINDOW_STEP_PIN, OUTPUT);
  pinMode(WINDOW_DIR_PIN, OUTPUT);
  digitalWrite(RELAY_PIN_QUAT, LOW); // Tắt relay ban đầu
  digitalWrite(RELAY_PIN_DEN, LOW); // Tắt relay ban đầu
  digitalWrite(RELAY_PIN_HEATER, LOW);
  digitalWrite(RELAY_PIN_MIST, LOW);
  digitalWrite(MOTOR_PIN, LOW);
  digitalWrite(WINDOW_STEP_PIN, LOW);
  digitalWrite(WINDOW_DIR_PIN, LOW);
}

void loop() {
  temp_sensor();

  if (Serial.available()) { 
    String data = Serial.readStringUntil('\n');
    data.trim();

    StaticJsonDocument<512> doc;
    DeserializationError error = deserializeJson(doc, data);

    String device = doc["device"];
    bool active = doc["active"];
    int power = doc["power"] | 0; // default 0

    Serial.print("[CMD] device=");
    Serial.print(device);
    Serial.print(" active=");
    Serial.print(active ? "true" : "false");
    Serial.print(" power=");
    Serial.println(power);

    if (configDevices(device, active, power)) {
      Serial.println("[OK] Command applied");
    } else {
      Serial.println("[ERR] Unknown device");
    }
  }

  delay(1000); // Đọc mỗi 1 giây
}

void temp_sensor() {
  float humidity = dht.readHumidity();
  float temperature = dht.readTemperature(); // Đọc nhiệt độ

  int gasValue = analogRead(gasPin); // Đọc giá trị từ cảm biến

  // Kiểm tra nếu đọc lỗi
  if (isnan(temperature) || isnan(humidity)) {
    return;
  }

  StaticJsonDocument<200> doc;
  doc["temperature"] = temperature;
  doc["humidity"] = humidity;
  doc["gas"] = gasValue;

  serializeJson(doc, Serial);
  Serial.println();
}

bool configDevices(String device, bool active, int power) {
  int fanPwm = map(constrain(power, 0, 100), 0, 100, 0, 255);

  if (device == "fan") {
    analogWrite(RELAY_PIN_QUAT, active ? fanPwm : 0);
  } else if (device == "light") {
    analogWrite(RELAY_PIN_DEN, active ? fanPwm : 0);
  } else if (device == "heater") {
    digitalWrite(RELAY_PIN_HEATER, active ? HIGH : LOW);
  } else if (device == "mist") {
    digitalWrite(RELAY_PIN_MIST, active ? HIGH : LOW);
  } else if (device == "feed") {
    digitalWrite(MOTOR_PIN, active ? HIGH : LOW);
  } else if (device == "window") {
    digitalWrite(WINDOW_DIR_PIN, active ? HIGH : LOW);
    for (int step = 0; step < WINDOW_STEPS_PER_MOVE; step++) {
      digitalWrite(WINDOW_STEP_PIN, HIGH);
      delayMicroseconds(800);
      digitalWrite(WINDOW_STEP_PIN, LOW);
      delayMicroseconds(800);
    }
  } else {
    return false;
  }

  return true;
}

void relay(int pin, int state ) {
  digitalWrite(pin, state);
}