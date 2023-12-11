#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include "serial_utils.h"

LiquidCrystal_I2C lcd(0x3f, 16,2); 
double entry_list[14];
double input_data[30][14];
uint8_t row = 0; 
 


void setup() {
  // put your setup code here, to run once:
  lcd.begin(); 
  lcd.backlight(); 
  lcd.clear(); 

  lcd.print("Ready for Data");

}

void loop() {
  
  get_data(entry_list, 14);
  set_row(entry_list, 14, input_data, row); 
  row++;
  
}
