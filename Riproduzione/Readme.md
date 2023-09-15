# Riproduzione *TynyML-based approach for Remaining Useful Life Prediction of Turbofan Engines*

L'obiettivo di questo lavoro è ricreare i modelli di proposti all'interno del paper di cui sopra i quali si propongono di calcolare il valore di **RUL** (*Remaining Useful Life*) per quanto riguarda dei motori a reazione proposti dalla **NASA** all'interno del dataset <a href="https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6">C-MAPSS Aircraft Engine Simulator Data</a>. 

## Descrizione del dataset

Data sets consists of multiple multivariate time series. Each data set is further divided into training and test subsets. Each time series is from a different engine i.e., the data can be considered to be from a fleet of engines of the same type. Each engine starts with different degrees of initial wear and manufacturing variation which is unknown to the user. This wear and variation is considered normal, i.e., it is not considered a fault condition. There are three operational settings that have a substantial effect on engine performance. These settings are also included in the data. The data is contaminated with sensor noise.

I data sets consistono di diverse serie temporali multivariate, ogni data set è poi diviso in una parte di *training* e una parte di *test*. Ogni serie temporale proviene da un diverso motore, di conseguenza i dati possono essere considerati come provenienti da una serie di motori della stessa tipologia. Ogni motore comincia con un diverso tipo di degrado iniziale e con alcune differenze a livello di fattura che non sono note all'utente (il degrado iniziale e le variazioni tra un motore e l'altro non sono considerate una condizione di *"fault"*). Ci sono tre *operational settings* le quali hanno un effetto sostanziale sulla performance del motore (queste vengono incluse nei dati). 

All'inizio di ogni serie temporale il motore opera normalmente e sviluppa un problema a un certo punto durante la serie. All'interno del *training set* il problema cresce fino a rendere il sistema inutilizzabile; nel *test set* la serie temporale si conclude prima che il sistema vada in stato di *failure*. 

L'obiettivo è quello di predirre il numero di cicli operazionali rimanenti prima che avvenga un fallimento per quanto riguarda i dati presenti nel test set. Per verificare l'inferenza della rete all'interno della cartella del dataset sono presenti i file *RUL_FD0001-4.txt* i quali contengono i valori della RUL per ogni unità operativa (per ogni motore)

Ogni file del dataset è un file di testo composto da 26 colonne di numeri separate da spazi. Ogni riga rappresenta lo snapshot dei dati raccolti durante un singolo ciclo operazionale, ogni colonna è una differente variabile. 

- Data Set: FD001
Train trjectories: 100
Test trajectories: 100
Conditions: ONE (Sea Level)
Fault Modes: ONE (HPC Degradation)


- Data Set: FD002
Train trjectories: 260
Test trajectories: 259
Conditions: SIX
Fault Modes: ONE (HPC Degradation)

- Data Set: FD003
Train trjectories: 100
Test trajectories: 100
Conditions: ONE (Sea Level)
Fault Modes: TWO (HPC Degradation, Fan Degradation)

- Data Set: FD004
Train trjectories: 248
Test trajectories: 249
Conditions: SIX
Fault Modes: TWO (HPC Degradation, Fan Degradation)

Ci sono 6 condizioni (o combinaizoni di esse) che possono verificarsi: 
- Condition 1: Altitude = 0, Mach Number = 0, TRA = 100
- Condition 2: Altitude = 10, Mach Number = 0.25, TRA = 100
- Condition 3: Altitude = 20, Mach Number = 0.7 TRA = 100
- Condition 4: Altitude = 25, Mach Number = 0.62, TRA = 60
- Condition 5: Altitude = 35 Mach Number = 0.84, TRA = 100
- Condition 6: Altitude = 42, Mach Number = 0.84, TRA = 100

Per quanto riguarda i datasets abbiamo: 

- FD001: **soltanto** condizione 1

- FD002: **Mix** di tutte le condizioni 

- FD003: **soltanto** condizione 1

- FD004: **Mix** di tutte le condizioni


Le colonne corrispondono a

1) unit number
2) time, in cycles
3) operational setting 1
4) operational setting 2
5) operational setting 3
6) sensor measurement 1
7) sensor measurement 2 
[...]
'26') sensor measurement 21

## La riproduzione

### Caricamento dei dati e prima elaborazione

Il caricamento dei dati in memoria viene effettuato utilizzando le librerie *Pandas* e *NumPy*, i dati vengono caricati in 4 dataframe i quali corrispondono ai vari FD0001-4.txt che si trovano all'interno del dataset. 
Essi vengono tenuti separati per semplicità di confronto con il paper di cui sopra. 
<figure>
<img src='/home/aliquodfahriam/tinyML/DrawIO/Diagramma senza titolo.drawio.png' id='train'> 

<figcaption align='center'><i>Rappresentazione grafica della lista train</i></figcaption>
</figure>


