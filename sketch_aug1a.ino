#include <Arduino.h>
#include <Wire.h>
#include <SoftwareSerial.h>
#include <MeMegaPi.h>
#include <MePS2.h>

// --- 基础硬件定义 ---
double angle_rad = PI/180.0;
double angle_deg = 180.0/PI;
MeEncoderOnBoard Encoder_1(SLOT1);
MeEncoderOnBoard Encoder_2(SLOT2);
MeEncoderOnBoard Encoder_3(SLOT3);
MeEncoderOnBoard Encoder_4(SLOT4);
MeMegaPiDCMotor motor_2(2);
MeMegaPiDCMotor motor_9(9);
MeMegaPiDCMotor motor_10(10);
MeGyro gyro_1;
MeMegaPiDCMotor motor_3(3);
MeMegaPiDCMotor motor_11(11);
MeMegaPiDCMotor motor_4(4);
Servo servo_6_1;
MePort port_6(6);
MeMegaPiDCMotor motor_12(12);
MePS2 MePS2(PORT_15);

// --- 硬件映射别名 ---
MeMegaPiDCMotor& motor_vert_right = motor_10;
MeMegaPiDCMotor& motor_vert_left  = motor_9;
MeMegaPiDCMotor& motor_pitch_mid  = motor_2;
MeMegaPiDCMotor& motor_yaw_right  = motor_3;
MeMegaPiDCMotor& motor_yaw_left   = motor_11;
MeMegaPiDCMotor& motor_arm_extend = motor_4;
MeMegaPiDCMotor& motor_arm_pitch  = motor_12;
Servo& servo_gripper = servo_6_1;

// --- 状态机定义 ---
enum Mode { TELEOP, AUTO };
Mode currentMode = TELEOP;

// --- AUTO模式所需的状态变量 ---
String serialData;
bool newData = false;
unsigned long lastCommandTime = 0;
const unsigned long FAILSAFE_TIMEOUT = 1500;
int grabCommand = 0;
int lastGrabCommand = 0;
bool isGrabbing = false;

// --- 函数声明 ---
void _loop();
void stopAllActuators();
void executeGrabSequence();
void readSerialData();
void parseAndExecuteCommands(String data);
String getValue(String data, String key);


void isr_process_encoder1(void){
  if(digitalRead(Encoder_1.getPortB()) == 0){
    Encoder_1.pulsePosMinus();
  }else{
    Encoder_1.pulsePosPlus();
  }
}

void isr_process_encoder2(void){
  if(digitalRead(Encoder_2.getPortB()) == 0){
    Encoder_2.pulsePosMinus();
  }else{
    Encoder_2.pulsePosPlus();
  }
}

void isr_process_encoder3(void){
  if(digitalRead(Encoder_3.getPortB()) == 0){
    Encoder_3.pulsePosMinus();
  }else{
    Encoder_3.pulsePosPlus();
  }
}

void isr_process_encoder4(void){
  if(digitalRead(Encoder_4.getPortB()) == 0){
    Encoder_4.pulsePosMinus();
  }else{
    Encoder_4.pulsePosPlus();
  }
}

void _delay(float seconds) {
  if(seconds < 0.0){
    seconds = 0.0;
  }
  long endTime = millis() + seconds * 1000;
  while(millis() < endTime) _loop();
}

void stopAllActuators() {
  motor_vert_right.run(0);
  motor_vert_left.run(0);
  motor_pitch_mid.run(0);
  motor_yaw_right.run(0);
  motor_yaw_left.run(0);
  motor_arm_extend.run(0);
  motor_arm_pitch.run(0);
  servo_gripper.write(0);
}

void readSerialData() {
  while (Serial.available() > 0) {
    char receivedChar = Serial.read();
    if (receivedChar == '<') {
      serialData = "";
    } else if (receivedChar == '>') {
      newData = true;
      return;
    } else {
      serialData += receivedChar;
    }
  }
}

String getValue(String data, String key) {
  String searchKey = key + ":";
  int keyIndex = data.indexOf(searchKey);
  if (keyIndex == -1) {
    return "0";
  }
  int startIndex = keyIndex + searchKey.length();
  int endIndex = data.indexOf(',', startIndex);
  if (endIndex == -1) {
    endIndex = data.length();
  }
  return data.substring(startIndex, endIndex);
}

void parseAndExecuteCommands(String data) {
  motor_vert_right.run(getValue(data, "vr").toInt());
  motor_vert_left.run(getValue(data, "vl").toInt());
  motor_pitch_mid.run(getValue(data, "pm").toInt());
  motor_yaw_right.run(getValue(data, "yr").toInt());
  motor_yaw_left.run(getValue(data, "yl").toInt());
  motor_arm_pitch.run(getValue(data, "ap").toInt());
  grabCommand = getValue(data, "ga").toInt();
}

void executeGrabSequence() {
  isGrabbing = true;
  Serial.println("--- GRAB SEQUENCE INITIATED ---");
  Serial.println("Step 1: Extending arm...");
  servo_gripper.write(30);
  motor_arm_extend.run(-127.5);
  _delay(2.5);
  
  Serial.println("Step 2: Closing gripper...");
  motor_arm_extend.run(0);
  servo_gripper.write(90);
  _delay(0.5);
  
  Serial.println("Step 3: Retracting arm...");
  motor_arm_extend.run(127.5);
  _delay(2.5);
  
  Serial.println("Step 4: Resetting...");
  motor_arm_extend.run(0);
  servo_gripper.write(0);
  Serial.println("--- GRAB SEQUENCE COMPLETE ---");
  isGrabbing = false;
}


void setup() {
  TCCR1A = _BV(WGM10);
  TCCR1B = _BV(CS11) | _BV(WGM12);
  TCCR2A = _BV(WGM21) | _BV(WGM20);
  TCCR2B = _BV(CS21);
  MePS2.begin(115200);
  gyro_1.begin();
  servo_6_1.attach(port_6.pin1());
  Serial.begin(115200);
  Serial.println("System Initialized. Starting in Teleop Mode.");
  currentMode = TELEOP;
}

void _loop() {
  MePS2.loop();
  gyro_1.update();
}


// ==================================================================
// --- 主循环 LOOP: 使用状态机模型 ---
// ==================================================================
void loop() {
  
  // --- 1. 状态切换逻辑 ---
  if (MePS2.ButtonPressed(7)) {
    if (currentMode == TELEOP) {
      currentMode = AUTO;
      stopAllActuators();
      Serial.println("Switched to AUTO Mode.");
      lastCommandTime = millis();
    }
  }
  if (MePS2.ButtonPressed(3)) {
    if (currentMode == AUTO) {
      currentMode = TELEOP;
      stopAllActuators();
      Serial.println("Switched to TELEOP Mode.");
    }
  }

  // --- 2. 根据当前模式执行对应任务 ---
  if (currentMode == TELEOP) {
    // --- 遥控模式代码 ---
    if((MePS2.ButtonPressed(5)) || (MePS2.ButtonPressed(1))){
      if(MePS2.ButtonPressed(5)){
        motor_2.run(-255);
        motor_9.run(255);
        motor_10.run(255);
        _delay(0.05);
      }
      if(MePS2.ButtonPressed(1)){
        motor_2.run(255);
        motor_9.run(-255);
        motor_10.run(-255);
        _delay(0.05);
      }
      motor_2.run(0);
      motor_9.run(0);
      motor_10.run(0);
    }
    if(gyro_1.getAngle(3) > 12){
      motor_10.run(178.5);
      _delay(0.05);
    }else{
      if(gyro_1.getAngle(2) < -12){
        motor_10.run(-178.5);
        _delay(0.05);
      }
      motor_10.run(0);
    }
    if(MePS2.ButtonPressed(14)){
      motor_3.run(255);
      motor_11.run(-255);
      _delay(0.05);
    }
    if(MePS2.ButtonPressed(15)){
      motor_3.run(-255);
      motor_11.run(255);
      _delay(0.05);
    }
    if(MePS2.ButtonPressed(16)){
      motor_3.run(-191.25);
      motor_11.run(-191.25);
      _delay(0.05);
    }
    if(MePS2.ButtonPressed(17)){
      motor_3.run(191.25);
      motor_11.run(191.25);
      _delay(0.05);
    }
    motor_3.run(0);
    motor_11.run(0);

    if(MePS2.ButtonPressed(12)){
      motor_4.run(-127.5);
      _delay(0.025);
    }
    if(MePS2.ButtonPressed(10)){
      motor_4.run(127.5);
      _delay(0.025);
    }
    motor_4.run(0);

    if(MePS2.ButtonPressed(9)){
      servo_6_1.write(90);
    }
    if(MePS2.ButtonPressed(11)){
      servo_6_1.write(0);
    }

    if(MePS2.MeAnalog(8) > 10){
      motor_12.run(127.5);
      _delay(0.025);
    }
    if(MePS2.MeAnalog(8) < -10){
      motor_12.run(-127.5);
      _delay(0.025);
    }
    motor_12.run(0);

  } else if (currentMode == AUTO) {
    // --- 自动模式代码 ---
    readSerialData();
    if (newData) {
      parseAndExecuteCommands(serialData);
      lastCommandTime = millis();
      newData = false;
    }

    if (grabCommand == 1 && lastGrabCommand == 0 && !isGrabbing) {
      executeGrabSequence();
    }
    lastGrabCommand = grabCommand;

    if (millis() - lastCommandTime > FAILSAFE_TIMEOUT) {
      Serial.println("FAILSAFE TRIGGERED: Communication lost. Stopping all actuators.");
      stopAllActuators();
      lastCommandTime = millis();
    }
  }

  // --- 3. 每次主循环最后都更新一次核心服务 ---
  _loop();
}