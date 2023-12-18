#include <Wire.h>
#include <LiquidCrystal_I2C.h> 

#include <TensorFlowLite.h>
//#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"
LiquidCrystal_I2C lcd(0x3f, 16,2); 
void populateInputVector(TfLiteTensor* input, int size) {
  Serial.begin(115200); 
  int counter = 0; 
  MicroPrintf("Popoliamo il vettore di input");
  while (true){
    //input->data.f[i] = input_data[i];
    if(Serial.available() > 0){
      String new_message = Serial.readStringUntil('\n');
      //SERIAL FLUSH
      while(Serial.available() > 0){
        char t = Serial.read(); 
      }

      if (counter < size){
        input->data.f[counter] = new_message.toFloat();
        Serial.println("OK\n"); 
        counter++; 
      }
      if(counter == size){
        break; 
      }

    }
  }
}
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr; //Input size 420
TfLiteTensor* output = nullptr;


constexpr int kTensorArenaSize = 16384;
// Keep aligned to 16 bytes for CMSIS
alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}  // namespace




void setup() {
  lcd.begin(); 
  lcd.backlight();
  lcd.clear(); 
  lcd.print("Ready"); 
  // put your setup code here, to run once:
  tflite::InitializeTarget();
  model = tflite::GetModel(small_lstm_batch_1_quad_tflite);  
  //static tflite::AllOpsResolver resolver;

  static tflite::MicroMutableOpResolver<3> resolver;
  resolver.AddUnidirectionalSequenceLSTM(); 
  resolver.AddReshape();
  resolver.AddFullyConnected();

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }
  input = interpreter->input(0);
  output = interpreter->output(0);

}

void loop() {
  populateInputVector(input, 420);
  if (kTfLiteOk != interpreter->Invoke()) {
    MicroPrintf("Invoke failed.");
  }
  float RUL = output->data.f[0];

  lcd.clear(); 
  lcd.print("RUL: "+ String(RUL)); 

}
