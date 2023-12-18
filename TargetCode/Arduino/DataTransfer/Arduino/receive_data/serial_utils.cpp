#include "serial_utils.h"

void get_data(double* entry_list, uint8_t size) {
  Serial.begin(115200);
  uint8_t counter = 0;

  while (true) {
    if (Serial.available() > 0) {
      String new_message = Serial.readStringUntil('\n');
      
      //SERIAL FLUSH
      while (Serial.available() > 0) {
        char t = Serial.read();
      }

      if (counter < size) {
        entry_list[counter] = new_message.toDouble();
        Serial.println("OK\n");
        //Serial.println(entry_list[counter], 17);
        counter++;  
      }

      if (counter == size) {
        break;  // Esci dal loop quando hai riempito l'array
      }
    }
  }
}