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

### Caricamento dei dati e prima elaborazione (1-4)

Il caricamento dei dati in memoria viene effettuato utilizzando le librerie *Pandas* e *NumPy*, i dati vengono caricati in 4 dataframe i quali corrispondono ai vari FD0001-4.txt che si trovano all'interno del dataset. 
Essi vengono tenuti separati per semplicità di confronto con il paper di cui sopra. 
<figure>
<img src='../DrawIO/Diagramma senza titolo.drawio.png' id='train'> 

<figcaption align='center'><i>Rappresentazione grafica della lista train</i></figcaption>
</figure>

#### Considerazioni sui sensori (5-15)
All'interno della sezione di *Data Preprocessing* del paper in esame viene detto:

>After reviewing all sensors, it is found that some of them
have constant values, hence we can remove them. Therefore,
14 sensors remained for each sub-dataset (sensor 2, 3, 4, 7,
8, 9, 11, 12, 12, 13, 14, 15, 17 and 20).

**Tuttavia** come appreso tramite l'analisi dei dati all'interno del training set ciò non è vero per ogni parte componente il dataset. 

È invece vero per quanto riguarda **FD0001 e FD0003**. La causa è probabilmente da ricercarsi in quanto detto pocanzi per quanto riguarda le condizioni di volo registrate dai sensori, si ricorda infatti che FD0001 e FD0003 rappresentano voli in condizioni **costanti** mentre FD0002 e FD0004 ne rappresentano altri in condizioni **miste**. 

Riportiamo i grafici che rappresentano i valori dei sensori rispettivamente in FD0001 e FD0002:

<figure>
<img src='../DrawIO/FD0001_sensors.png'>
<figcaption align='center'><i>Sensori FD0001</i></figcaption>
</figure>

<figure>
<img src='../DrawIO/FD0002_sensors.png'>
<figcaption align = 'center'><i>Sensori FD0002</i></figcaption>
</figure>

Gli autori del paper hanno preso la decisione di eliminarli da ogni componente del dataset. La *ratio* che li ha portati a questa decisione non si evince dal documento, tuttavia si potrebbe ipotizzare che i valori di quei sensori siano costanti per ogni condizione e che quindi rappresentino una *"sommatoria di tratti costanti a seconda della condizione di volo"*, ma questa rimane comunque soltanto un'ipotesi. 

Per correttezza nella riproduzione (e per rendere i risultati maggiormente comparabili) abbiamo scelto di rimuovere gli stessi sensori anche noi. 

I sensori rimossi sono: 
- Sensor 1
- Sensor 5 
- Sensor 10 
- Sensor 16
- Sensor 18
- Sensor 19

### Min Max Scaling (16-24)

Dato che i sensori sono affetti da rumore si è deciso di applicare del Min Max scaling. Ciò viene effettuato tramite l'oggetto **MinMaxScaler** presente nella libreria *sklearn* il quale si occuperà di effettuare l'operazione su ogni dataframe istanziato all'interno della lista *train*. 
Per sicurezza abbiamo verificato che la distribuzione dei dati non fosse cambiata a seguito di qualche errore nel codice. Riportiamo  di seguito i grafici delle distribuzioni dei dati dei vari sensori prima e dopo lo scaling: 

<figure>
<img src='../DrawIO/sensor_density_default.png'>
<figcaption align='center'><i>Distribuzione FD0001 pre-scaling</i></figcaption>
</figure>

<figure>
<img src='../DrawIO/sensor_density_scaled.png'>
<figcaption align='center'><i>Distribuzione FD0001 post-scaling</i></figcaption>
</figure>

NB:*Tutti i grafici sono stati realizzati tramite le librerie Seaborn e Matplotlib di Python*

