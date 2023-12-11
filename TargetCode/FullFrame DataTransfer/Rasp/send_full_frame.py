import numpy as np 
import serial 
import time
#Carico i dati 
processed_test_data = np.float32(np.load("../processed_test_data_np.npy")) 
data = processed_test_data[:1,:1, :14]
#Imposto la porta seriale 

ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1.0 )
ser.reset_input_buffer()
print("Serial OK" )

#Ottengo comunque un array di 3 dimensioni anche se ho preso soltanto la prima riga 
print(data)
data = data.flatten()#Riduco l' array a una singola dimensione
try:
	for j in range(1,31):
		data = processed_test_data[:1, j-1:j,:14]
		data = data.flatten(); 
		for i in data:
			print("sending: ", i);
			ser.write((str(i)+"\n").encode('utf-8'))
			while True:
				time.sleep(0.01);
				if ser.in_waiting > 0:
					response = ser.readline().decode('utf-8').rstrip()
					print(response)
					if  response == "OK":
						ser.flushInput(); 
						ser.flushOutput(); 
						print("Ricevuto OK")
						break
					else:
						ser.write((str(i)+"\n").encode('utf-8'))
			time.sleep(0.150)
except KeyboardInterrupt:
	ser.close()
	print("Serial connection closed")

ser.close()
print("Serial connection closed")				
