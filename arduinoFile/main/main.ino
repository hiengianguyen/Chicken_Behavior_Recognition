#include "DHT.h"
#include "ArduinoJson.h"

#define DHTPIN A1        // Chân DATA của DHT11
#define DHTTYPE DHT11   // Loại cảm biến

#define RELAY_PIN_QUAT 3     // Chân điều khiển relay
#define RELAY_PIN_DEN 5     // Chân điều khiển relay
#define MOTOR_PIN 8     // Chân điều khiển đồ ăn tự động

const int gasPin = A0; // Chân analog nối với module gas
const int threshold = 400; // Mức giới hạn cảnh báo (tùy chỉnh)

int sensorValue = 0;      // Biến lưu giá trị đọc được

DHT dht(DHTPIN, DHTTYPE);

void setup() {
  Serial.begin(9600);
  dht.begin();

  pinMode(RELAY_PIN_QUAT, OUTPUT);
  pinMode(RELAY_PIN_DEN, OUTPUT);
  digitalWrite(RELAY_PIN_QUAT, LOW); // Tắt relay ban đầu
  digitalWrite(RELAY_PIN_DEN, LOW); // Tắt relay ban đầu
}

void loop() {
  temp_sensor();

  // if (Serial.available()) { 
  //   String data = Serial.readStringUntil('\n');
  //   data.trim();

  //   StaticJsonDocument<512> doc;
  //   DeserializationError error = deserializeJson(doc, data);

  //   if (error) {
  //     Serial.println("JSON ERROR");
  //     return;
  //   }

  //   String device = doc["device"];
  //   bool active = doc["active"];
  //   int power = doc["power"] | 0; // default 0

  //   configDevices(device, active, power);
  // }

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

void configDevices(String device, bool active, int power) {
  if (device == "fan") {
      if (active) {
        int pwm = map(power, 0, 100, 0, 255);
        analogWrite(RELAY_PIN_QUAT, pwm);
      } else {
        analogWrite(RELAY_PIN_QUAT, 0);
      }
    } 
    else if (device == "light") {
      if (active) {
        int pwm = map(power, 0, 100, 0, 255);
        analogWrite(RELAY_PIN_DEN, pwm);
      } else {
        analogWrite(RELAY_PIN_DEN, 0);
      }
    } 
    else if (device == "feed") {
      if (active) {
        analogWrite(MOTOR_PIN, 255);
      } else {
        analogWrite(MOTOR_PIN, 0);
      }
    }
}

void relay(int pin, int state ) {
  digitalWrite(pin, state);
}