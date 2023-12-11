#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include "serial_utils.h"

LiquidCrystal_I2C lcd(0x3f, 16,2); 
double entry_list[14];
uint8_t counter = 0; 


void setup() {
  // put your setup code here, to run once:
  lcd.begin(); 
  lcd.backlight(); 
  lcd.clear(); 

  Serial.begin(115200);
  lcd.print("Ready for Data");

}

void loop() {
  
  
  get_data(entry_list, 14);

  
  
  for(uint8_t i = 0; i < 14 ; i++){
    lcd.clear(); 
    lcd.print(String(entry_list[i])+" "+i);
    delay(2000);
  }
  
 
}
