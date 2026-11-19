#include "DHT.h"
#include "ArduinoJson.h"
#include <Servo.h>

// Pin và loại cảm biến khai báo bằng macro #define.
#define DHTPIN A1                 // Chân DATA của DHT11
#define DHTTYPE DHT11             // Loại cảm biến
#define RELAY_PIN_QUAT 3          // Relay quạt
#define RELAY_PIN_DEN 5           // Relay đèn
#define RELAY_PIN_HEATER 6        // Relay sưởi
#define RELAY_PIN_MIST 7          // Relay phun sương
#define MOTOR_PIN 8               // Động cơ cấp thức ăn
#define WINDOW_SERVO_PIN 9        // Servo cửa sổ
#define FLAME_SENSOR_PIN 2        // Chân DO của cảm biến lửa
#define BUZZER_PIN 4              // Còi báo

// Hằng số có kiểu dữ liệu, khai báo bằng const.
const int GAS_PIN = A0; // Chân analog của cảm biến gas
const int WINDOW_SERVO_CLOSED_ANGLE = 0;
const int WINDOW_SERVO_OPEN_ANGLE = 90;
const unsigned long WINDOW_MOVE_INTERVAL_MS = 20;
const unsigned long SENSOR_INTERVAL_MS = 1000;
const bool FLAME_SENSOR_ACTIVE_LOW = true;

int windowCurrentPercent = 0; // Giả sử cửa sổ bắt đầu ở vị trí đóng.
int windowTargetPercent = 0;
bool fireAlarmSilenced = false;
bool fireDetectedCurrent = false;
unsigned long lastWindowMoveMs = 0;
unsigned long lastSensorReadMs = 0;

DHT dht(DHTPIN, DHTTYPE);
Servo windowServo;

void setup() {
  Serial.begin(9600);
  Serial.setTimeout(50);
  dht.begin();

  pinMode(RELAY_PIN_QUAT, OUTPUT);
  pinMode(RELAY_PIN_DEN, OUTPUT);
  pinMode(RELAY_PIN_HEATER, OUTPUT);
  pinMode(RELAY_PIN_MIST, OUTPUT);
  pinMode(MOTOR_PIN, OUTPUT);
  pinMode(FLAME_SENSOR_PIN, INPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(RELAY_PIN_QUAT, LOW);
  digitalWrite(RELAY_PIN_DEN, LOW);
  digitalWrite(RELAY_PIN_HEATER, LOW);
  digitalWrite(RELAY_PIN_MIST, LOW);
  digitalWrite(MOTOR_PIN, LOW);
  digitalWrite(BUZZER_PIN, LOW);

  windowServo.attach(WINDOW_SERVO_PIN);
  writeWindowServoAngle(windowCurrentPercent);
}

void loop() {
  unsigned long now = millis();
  updateFireAlarmState();

  if (now - lastSensorReadMs >= SENSOR_INTERVAL_MS) {
    publishSensorData();
    lastSensorReadMs = now;
  }

  if (Serial.available()) {
    String data = Serial.readStringUntil('\n');
    data.trim();

    StaticJsonDocument<512> doc;
    DeserializationError error = deserializeJson(doc, data);
    if (error) {
      Serial.print("[ERR] Invalid JSON: ");
      Serial.println(error.c_str());
      updateWindowServo();
      return;
    }

    String device = doc["device"];
    bool active = doc["active"];
    int power = doc["power"] | 0;

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

  updateWindowServo();
}

void publishSensorData() {
  updateFireAlarmState();

  float humidity = dht.readHumidity();
  float temperature = dht.readTemperature();
  int gasValue = analogRead(GAS_PIN);

  StaticJsonDocument<256> doc;
  if (!isnan(temperature)) {
    doc["temperature"] = temperature;
  }
  if (!isnan(humidity)) {
    doc["humidity"] = humidity;
  }
  doc["gas"] = gasValue;
  doc["fire_detected"] = fireDetectedCurrent;
  doc["fire_alarm_silenced"] = fireAlarmSilenced;
  doc["fire_alarm_active"] = fireDetectedCurrent && !fireAlarmSilenced;

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
    moveWindowToPercent(active ? power : 0);
  } else if (device == "fire_alarm") {
    fireAlarmSilenced = !active;
    updateFireAlarmState();
  } else {
    return false;
  }

  return true;
}

void updateFireAlarmState() {
  bool fireDetected =
      digitalRead(FLAME_SENSOR_PIN) == (FLAME_SENSOR_ACTIVE_LOW ? LOW : HIGH);

  if (fireDetected && !fireDetectedCurrent) {
    fireAlarmSilenced = false;
  } else if (!fireDetected) {
    fireAlarmSilenced = false;
  }
  fireDetectedCurrent = fireDetected;

  digitalWrite(
      BUZZER_PIN,
      fireDetectedCurrent && !fireAlarmSilenced ? HIGH : LOW);
}

void moveWindowToPercent(int targetPercent) {
  windowTargetPercent = constrain(targetPercent, 0, 100);
}

void updateWindowServo() {
  unsigned long now = millis();
  if (windowCurrentPercent == windowTargetPercent ||
      now - lastWindowMoveMs < WINDOW_MOVE_INTERVAL_MS) {
    return;
  }

  lastWindowMoveMs = now;
  windowCurrentPercent += windowTargetPercent > windowCurrentPercent ? 1 : -1;
  writeWindowServoAngle(windowCurrentPercent);
}

void writeWindowServoAngle(int percent) {
  int angle = map(
      percent,
      0,
      100,
      WINDOW_SERVO_CLOSED_ANGLE,
      WINDOW_SERVO_OPEN_ANGLE);
  windowServo.write(constrain(angle, 0, 180));
}