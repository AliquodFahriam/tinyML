# RUL prediction based on a double CNN

In questo caso vediamo l'utilizzo di una doppia CNN a differenza di quanto detto dalla review la quale consiglia per questi scopi l'utilizzo di **RNN** e, in particolare, di una loro sottospecie, ovvero le **LSTM**

**Dataset:** Pronostia Bearing Dataset 

Un passo avanti che questo documento mostra è il riconoscimento di diversi pattern di degrado e mette in luce l'importanza del determinare dinamicamente da questi una soglia per cui la RUL possa essere calcolata in maniera efficace. 

Viene quindi proposta una architettura basata su doppia CNN profonda in cui la prima rete si occupa di identificare il punto in cui porre la soglia mentre la seconda si occupa di predirre la RUL. 

![](/home/aliquodfahriam/debianHome/.config/marktext/images/2023-07-24-15-30-05-image.png)

## Modello di classificazione per l'identificazione dei punti IF (Incipient Failure)

I tipi di degrado sono principalmente 2: 

* Degrado rapito 

* Degrado lento

Le reti vengono addestrate come segue: 

1) Il segnale di vibrazione viene classificato in *healthy* o *faulty* 

2) Questi segnali sono utilizzati per addestrare la rete 

A questo punto si propongono i dati di test alla rete in maniera che questi li classifichi in maniera corretta come faulty o healthy. Nel caso vengano trovati 3 punti faulty su 5 analizzati consecutivamente dalla rete allora si è in presenza di un punto di **IF**. 

## Modello di regressione tra il segnale originale e la percentuale di *reliability*

![](/home/aliquodfahriam/debianHome/.config/marktext/images/2023-07-24-15-45-05-IMG_0182.jpeg)

Viene qui proposto un meccanismo di mapping tra RUL e reliability (R) la quale viene calcolata come segue: 

$R(i) = 1 $  se  $T_i  \le T_1$

$R(i) = 1 - \frac {T_i - T_1} {T_e -T_1}$  se $T_i \gt T_1$

Per i dati di training ovviamente il valore di $T_e$ risulta noto. $R(i)$ è la percentuale di reliability dei dati di training al tempo $T_i$. $T_e$ è il tempo totale del test; $T_1$ è il tempo di partenza per l'incipient failure. 



## Descrizione del dataset PRONOSTIA

The experimental data come from the publicly available PRONOSTIA platform, which has been widely used to verify the effectiveness of RUL prediction methods [32]. During  the experiments, a radial force of 4 kN is applied on the test  bearings to conduct accelerated life tests. The rotating speed  of bearings is set as 1800 r/min. Two vibration sensors with  25.6 kHz sampling frequency are mounted on the bearing to monitor the degradation process: one is set on the vertical axis, and the other one is on the horizontal axis. In this paper, the vertically collected signals are used for analysis. The length of every sample is 0.1 s with 2560 points, and the sampling is repeated every 10 s. All tests are stopped when the amplitude of  the vibration signal exceeds 20 g. The experimental system, as well as the tested bearings before and after a test, can be found in [32]



### Altri dati utili:

Uno dei dati che vengono spesso utilizzati all'interno di questo documento è l'**RMS**, ma di che cosa si tratta? 

#### Accelerometer Noise Density

È la radice quadrata della densità spettrale di potenza del rumore. Il rumore totale viene calcolato come:
$ Noise = NoiseDensity * \sqrt {BW * 1.6}$

Dove BW sta per la banda dell'accelerometro. 


