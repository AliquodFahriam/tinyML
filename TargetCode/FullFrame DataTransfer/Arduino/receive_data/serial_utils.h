#ifndef SERIAL_UTILS_H
#define SERIAL_UTILS_H
#include <Arduino.h>
#include <string>
#include <iostream>

void get_data(double* entry_list, uint8_t size); 
void set_row(double* entry_list, uint8_t size, double input_data[30][14], uint8_t counter); 

#endif
