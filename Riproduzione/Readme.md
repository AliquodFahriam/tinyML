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

### Modelli

Gli autori del paper che stiamo cercando di riprodurre propongono come migliori modelli due reti **LSTM** (Long Short Term memory) una di dimensioni ridotte mentre l'altra full size, esse sono rappresentate come segue:

Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param 
=================================================================
 lstm_4 (LSTM)               (None, 30, 60)            18240     
                                                                 
 lstm_5 (LSTM)               (None, 30, 30)            10920     
                                                                 
 flatten_1 (Flatten)         (None, 900)               0         
                                                                 
 dense_3 (Dense)             (None, 30)                27030     
                                                                 
 dense_4 (Dense)             (None, 15)                465       
                                                                 
 dense_5 (Dense)             (None, 1)                 16        
                                                                 
=================================================================
Total params: 56,671
Trainable params: 56,671
Non-trainable params: 0
_________________________

Layer (type)                Output Shape              Param #   
=================================================================
 lstm (LSTM)                 (None, 30, 128)           73728     
                                                                 
 dropout (Dropout)           (None, 30, 128)           0         
                                                                 
 lstm_1 (LSTM)               (None, 30, 64)            49408     
                                                                 
 dropout_1 (Dropout)         (None, 30, 64)            0         
                                                                 
 lstm_2 (LSTM)               (None, 30, 32)            12416     
                                                                 
 dropout_2 (Dropout)         (None, 30, 32)            0         
                                                                 
 lstm_3 (LSTM)               (None, 30, 16)            3136      
                                                                 
 dropout_3 (Dropout)         (None, 30, 16)            0         
                                                                 
 flatten (Flatten)           (None, 480)               0         
                                                                 
 dense (Dense)               (None, 64)                30784     
                                                                 
 dense_1 (Dense)             (None, 32)                2080      

Total params: 171,585
Trainable params: 171,585
Non-trainable params: 0

durante la nostra riproduzione verranno chiamate LSTMsmallModel e LSTMlargeModel.
Verranno inoltre valutate tramite RMSE e tramite una funzione asimmetrica come metriche per il punteggio. 

Come iperparametri troviamo: 
- Optimizer: Adam 
- Initial learning rate: 0.1 
- Epochs: 50, 100, 80, 150
- Batch size: 256
- Alpha value: 

***Funzione di Score:***
$$
Score = \begin{cases} \sum^{N}_{n=1} e^{\frac {-d_i} {13}} - 1   , & \text{if} \space \space d_i \lt 0 \\
\sum^{N}_{n=1} e^{\frac {d_i} {10}} - 1, &\text{otherwise} \space \space \end{cases}
$$

<figure>
<img src='../DrawIO/scoring_function.png'></img>
<figcaption align= 'center'>Score value as the error increases. The score is
calculated using the scoring function where late predictions receive higher penalisation </figcaption>
</figure>


***RMSE:***

$$
RMSE = \sqrt{\frac{1}{N}\sum^N_{i=1} d_i^2}
$$

Vengono inoltre applicate dagli autori delle ottimizzazioni ulteriori per quanto riguarda i modelli di cui sopra ai fini di una minor richiesta di risorse, il che risulta cruciale ai fini di un approccio di tipo TinyML. 

I modelli sono stati sviluppati utilizzando la libreria *Tensorflow* di python e il dataset è stato diviso utilizzando il criterio 80/20, il che significa che l'80% dei dati di training è stato utilizzato per l'addestramento mentre la parte rimanente è stata utilizzata per la *validazione* in fase di addestramento.

### Considerazioni post addestramento dei modelli

Abbiamo deciso inizialmente di addestrare i modelli su tutti i dati disponibili contemporaneamente tuttavia dal paper che stiamo cercando di riprodurre sembra che l'addestramento dei modelli sia stato reiterato per ognuna delle componenti del dataset in maniera separata. 

Ovvero ad esempio la rete LSTM *"piccola"* è stata addestrata prima sui dati di FD0001, poi su quelli di FD0002 e così via... 

Ciò a primo impatto è sembrato strano e si è tentato un approccio che cercasse di fare in modo che tutti i dati contemporaneamente fossero utilizzabili per l'addestramento, tuttavia ciò non ha portato a risultati soddisfacenti, di conseguenza proveremo a creare reti addestrate specificamemnte sulle varie componenti del datast. 

### Proposte per il miglioramento 

Una proposta per il miglioramento di questo tipo di condizione potrebbe essere quella di dividere il dataset in due parti, da un lato avremmo le due componenti a condizioni costanti, ovvero: 
- FD0001 
- FD0003

mentre dall'altra avremmo le due componenti in condizioni miste: 
- FD0002
- FD0004

A questo punto addestrare su questi due gruppi di dati due reti separate e, in seguito creare un'ultima rete (un **classificatore**), il quale dovrebbe avere lo scopo di indirizzare i dati verso la rete che è più adatta a calcolare la RUL. 
Questa idea tuttavia presenta dei problemi di fondo che riguardano l'input delle reti di tipo LSTM. 

Una rete di questo tipo ha bisogno di un input del seguente tipo: 
- Campioni
- *Time Steps* 
- *Features* 

I time steps sarebbero il numero di campioni per ogni sequenza di dati passati alla rete. Ciò significa che il classificatore dovrebbe analizzare i dati in batch da *time steps* elementi e classificarli tutti insieme allo stesso modo il che potrebbe comportare degli errori e dei problemi. 

Nel paper che stiamo cercando di riprodurre la dimensione del vettore di input alla rete è la seuente: 
(batch_size, 30, 14). 

### Aggiornamenti 18/09/23

I risultati che otteniamo non si avvicinano per nulla con quelli ottenuti dagli autori del paper che stiamo cercando di riprodurre, ciò ci porta a pensare che il modo in cui elaboriamo i dati prima di darli in input alle reti neurali non sia ottimale, in particolare abbiamo: 

> Last cycles of an engine life are more significant than the initial cycles. Thus, a piecewise linear RUL function is applied, where a max RUL value is set if the true RUL is greater than LSTM this max value, as shown in Eq. 2. In this way, we ignore data whose true RULs are greater than the maximum limit to pay attention to the degradation data and we adopt a max RUL value of **125**, as used in [<a href='https://ieeexplore.ieee.org/abstract/document/8998569?casa_token=m6jBHKB0s_4AAAAA:qt23ITJDbDXkIeKp7wxjpRmv3OlJRg3otquWxHHAr1zO_nx1AwkSb5kNiqVw0S5gB_tAFjuv'>19</a>] and other related works.

Per di più anche la funzione di loss deve essere modificata in favore di una che sia maggiormente adatta alle nostre esigenze rispetto all'RMSE. In particolare abbiamo: 
$$
    Loss = \begin{cases} 2ad_i , & \text{if} \space \space d_i \lt 0 \\
    2(a+(1-2a))d_i, &\text{otherwise} \space \space \end{cases}
$$

Per come viene descritta nel seguente paper: <a href = 'https://ieeexplore.ieee.org/abstract/document/9207051?casa_token=mj5ETeDbMFIAAAAA:ZCf8jWyvO0wN6k7igZNQtXoMJGq5dSqb7YYiaeHxqL7M5L0Y1jkyrk8HzGxoq3_bnmy7tOHI'>Asymmetric loss
functions for deep learning early predictions of remaining useful life in
aerospace gas turbine engines,</a>

Una funzione asimmetrica di questo tipo è necessaria ai fini di penalizzare le predizioni che vanno oltre la RUL effettiva.

Una volta effettuate le precedenti modifiche bisogna comprendere il motivo per cui non viene implementata correttamente la funzione di score custom che abbiamo implementato 

#### Il problema della funzione di loss: 
Ovviamente la funzione di loss per come scritta sopra non potrebbe funzionare correttamente. Posto $d_i$ come la differenza tra il valore predetto e il valore corretto di RUL ($d_i = y_{pred} - y_{true}$) nel caso in cui y_true fosse maggiore ci ritroveremmo ad avere un valore della funzione di loss negativo, il che sarebbe fuorviante per l'apprendimento della rete. 

Ci teniamo a precisare che la funzione scritta sopra è riportata esattamente per come scritta all'interno dell'articolo, il quale tuttavia fa riferimento ad un <a href = 'https://ieeexplore.ieee.org/abstract/document/9207051?casa_token=mj5ETeDbMFIAAAAA:ZCf8jWyvO0wN6k7igZNQtXoMJGq5dSqb7YYiaeHxqL7M5L0Y1jkyrk8HzGxoq3_bnmy7tOHI'>secondo documento</a> il quale la riporta in questa maniera: 

$$
    Loss = \begin{cases} 2a(\hat y_i - y_i )^2 , & \text{if} \space \space d_i \lt 0 \\
    2(a+(1-2a))(\hat y_i - y_i )^2, &\text{otherwise} \space \space \end{cases}
$$

La ovvia differenza è data dai quadrati i quali risolvono il problema descritto sopra. 

Il documento di cui sopra non si limita a descrivere l'utilizzo di questa specifica funzione di loss ma fornisce innanzitutto un discreto *background* sulle funzioni di loss asimmetriche le quali possono risultare estremamente utili per quanto riguarda l'addestramento di modelli con l'obiettivo di calcolare la vita rimanente utile di un elemento in condizioni critiche. 

In particolare, la funzione di cui sopra è definita come *Quadratic-Quadratic*(QUAD - QUAD) ma si fa riferimento anche a *Mean Square Logarithmic Error-Mean Square Error* (MSLE-MS), *Linear-Mean Square Error* (LIN-MSE), Linear-Linear(LIN-LIN) le quali potrebbero risultare utili in altri contesti applicativi e magari con altri set di dati.


### Aggiornamenti 20/09/23

A seguito dell'addestramento delle due reti LSTM che abbiamo descritto in precedenza per quanto riguarda la componente FD0001 del datast abbiamo ottenuto i seguenti risultati per quanto riguarda la funzione di loss: 
- Small LSTM: 2912.22192
- Large LSTM: 6115.06201

Questi risultati non sono ancora soddisfacenti poiché, come riportato all'interno del documento <a href = 'https://ieeexplore.ieee.org/abstract/document/9207051?casa_token=mj5ETeDbMFIAAAAA:ZCf8jWyvO0wN6k7igZNQtXoMJGq5dSqb7YYiaeHxqL7M5L0Y1jkyrk8HzGxoq3_bnmy7tOHI'>Asymmetric loss
functions for deep learning early predictions of remaining useful life in
aerospace gas turbine engines</a>, il nostro target dovrebbe attestarsi intorno a **1647.3**

Nonostante la distanza si aggiri intorno al 10%, e di conseguenza i risultati siano tutto sommato comparabili, ciò è vero soltanto per la *Small LSTM*, il che è in controtendenza rispetto ai risultati ottenuti nel paper che stiamo cercando di riprodurre. Una separazione così drastica non è giustificabile e, prima di procedere ulteriormente con un confronto più approfondito alla luce delle metriche che gli autori hanno utilizzato ho deciso di mettere in luce alcune criticità della nostra particolare applicazione.

#### La data preparation
Il processo di data preparation si svolge come segue: 
1. Vengono raccolte le varie componenti del dataset in data frames
2. Vengono rimossi i sensori "*inutili*"
3. **Vengono divisi i dati**
3. Vengono formattati per essere dati in input durante la fase di addestramento. 

Risulta particolarmente critico il punto 3.

Cito testualmente: 
> Thus, a piecewise linear RUL function is applied,
where a max RUL value is set if the true RUL is greater than
this max value, as shown in Eq. 2. In this way, we ignore data
whose true RULs are greater than the maximum limit to pay
attention to the degradation data and we adopt a max RUL
value of 125, as used in [19] and other related works

Una tale funzione non è stata da noi implementata poiché non risultava chiaro il suo funzionamento e il proposito che perseguiva.
Ciò è stato chiarito dal documento: <a href = 'https://ieeexplore.ieee.org/abstract/document/9207051?casa_token=mj5ETeDbMFIAAAAA:ZCf8jWyvO0wN6k7igZNQtXoMJGq5dSqb7YYiaeHxqL7M5L0Y1jkyrk8HzGxoq3_bnmy7tOHI'>Asymmetric loss
functions for deep learning early predictions of remaining useful life in
aerospace gas turbine engines</a> il qale afferma: 

>Furthermore, the value of the maximum cycle s capped at 100 and remained constant until degradation has
occur as shown in Fig 2. This allows the deep learning models
to differentiate between the healthy state (RUL = 100) and
unhealthy state (RUL < 100). Even though degradation can
happen randomly, the early stages of engine cycle are assumed
to be usable and functional. The labels are the RUL cycle for
each instance of the data.

<figure>
<img src='../DrawIO/capped_rul.png'>
<figcaption align='center'>Maximum RUL of gas turbine engine are capped to
100 cycle to distinguish the healthy state and degradation state
during preprocessing stage.</figcaption>
</figure>

Nel caso del paper che stiamo cercando di riprodurre questo valore andrebbe settato a 125. 

Proseguendo in questa direzione inoltre una ulteriore accortezza va rispettata: 

>Finally, after the aforementioned data preprocessing steps,each train set is splitted into the following datasets: train (for ML model building) and validation (for ML model to validateon unseen data) with a 80/20% split, **with caution that the segment of each engine is not separated**

Nonostante la maggior parte del paragrafo precedente sia stata rispettata, non abbiamo tuttavia tenuto conto dell'importanza nel non separare i dati dello stesso motore, i quali, invece, sicuramente si "mescolano" seguendo l'algoritmo da noi proposto. 

Il prossimo passo quindi è quello di comprendere come elaborare i dati nella maniera in cui gli autori hanno pensato di elaborarli poiché non è l'unico lavoro che fa riferimento a questo tipo di data preparation per questo dataset. 

### Aggiornamenti 26/09/2023 

La funzione per il calcolo della RUL in fase di training (ovvero per calcolare i label dei valori del dataset) sembra essere qualcosa di universalmente accettato lavorando su questo dataset. 

Per quanto riguarda l'applicazione pratica, procediamo come segue: 
- Innanzitutto creiamo un nuovo notebook in cui ricaricare il tutto, così da conservare i risultati del vecchio che potrebbero essere utili

- Dopodiché implementiamo la funzione lineare a tratti facendo in modo che, una volta trattati i dati che devono essere di dimensione (*batch_size*, 30, 14) il valore corrispettivo della RUL sia al massimo 100 e cominci a diminuire quando effettivamente supera il valore desiderato. (Quindi nei primi 100 cicli il valore della RUL sarà sempre 100 piuttosto che valori superiori). 

L'unico problema rimane sempre quello di fare in modo che, durante la selezione dei campioni facenti parte di una sequenza, essi facciano sempre riferimento alla stessa unità e non ne vengano mischiate di successive. 

<figure>
<img src='../DrawIO/window_problem.drawio.png'>
<figcaption align='center'>Problema della finestra con diversi unit number</figsize>
</figure>

<a href = 'https://github.com/biswajitsahoo1111/rul_codes_open/tree/master'>Link</a>

#### La *piecewise-linear function*

La funzione *process_targets* all'interno del file <a href='Riproduzione_PW/riproduzione_pw.ipynb'>*riproduzione_pw*</a> è la cosiddetta funzione **lineare a tratti** di cui parliamo estensivamente all'interno del readme ci permette dunque di calcolare il valore della RUL da assegnare ad ogni elemento del dataset. 
La funzione prende in input la lunghezza totale dei dati e la *"early_rul"* la quale rappresenta il valore massimo possibile di RUL (imponiamo ciò poiché come descritto dalla letteratura dovrebbe permettere alla rete di comprendere meglio quando il componente è in salute o meno), il quale poi viene decrementato in maniera lineare quando la *"data_length"* supera il valore di *early_rul*. La funzione che abbiamo appena descritto si trova all'interno della cella *[4]* del file linkato in precedenza.

#### La funzione di *generazione delle sequenze* 

Come abbiamo già abbondantemente sottolineato, per poter lavorare con sequenze temporali di questo tipo e, soprattutto, per poterlo fare con reti di tipo LSTM, abbiamo bisogno di dividere i dati in sequenze da $n$ elementi. In questo caso, come da istruzioni (e da stato dell'arte) abbiamo posto un valore per la lunghezza delle sequenze di 30 elementi. 

La funzione che fa ciò si chiama *process_input_data_with_targets*, la quale crea le strutture dati necessarie a contenere i dati nella forma corretta, ovvero sequenze da 30 elementi con passo 1. Stavolta tutto il procedimento è fatto in maniera manuale piuttosto che con la funzione *pad_sequences* di *keras*
146.14778

### Aggiornamenti 2/10/23

#### LSTM Small
Siamo riusciti a creare un modello di small LSTM che abbia delle metriche abbastanza soddisfacenti e paragonabili con quelle che sono state riportate dal paper che stiamo cercando di riprodurre. 
Il modello small LSTM a, seguito delle modifiche apportate al codice di preparazione dei dati, è riuscito ad ottenere un valore per la funzione di loss pari a: loss: 156.5530, il che è ottimo, soprattutto se messo in correlazione con il documento <a href = 'https://ieeexplore.ieee.org/abstract/document/9207051?casa_token=mj5ETeDbMFIAAAAA:ZCf8jWyvO0wN6k7igZNQtXoMJGq5dSqb7YYiaeHxqL7M5L0Y1jkyrk8HzGxoq3_bnmy7tOHI'>Asymmetric loss
functions for deep learning early predictions of remaining useful life in
aerospace gas turbine engines</a> in cui il valore della funzione di loss perquesto stesso tipo di rete era pari a 1647.3.

Attenzione però, i dati di cui siamo in possesso al momento riguardano esclusivamente la fase di training e validation, non di testing. 
Tuttavia il valore della funzione di loss in fase di validazione può essere un buon metro per comprendere se la rete ha generalizzato in maniera corretta o meno e, in questo caso il valore della Loss in fase di validazione è di 146.14 per la nostra LSTM_small.

Adesso andranno condotti test utilizzando RMSE e l'S score come definiti pocanzi, ovvero: 

$$
S \space score = \begin{cases} \sum^{N}_{n=1} e^{\frac {-d_i} {13}} - 1   , & \text{if} \space \space d_i \lt 0 \\
\sum^{N}_{n=1} e^{\frac {d_i} {10}} - 1, &\text{otherwise} \space \space \end{cases}
$$

$$
RMSE = \sqrt{\frac{1}{N}\sum^N_{i=1} d_i^2}
$$

per poter avere un preciso confronto con i valori ottenuti 

#### LSTM large 
Per quanto riguarda questo tipo di rete abbiamo dovuto modificare abbondantemente il codice. Innanzitutto presentiamo la specifica configurazione utilizzata per ottenere i risultati esposti in precedenza per la rete **LSTM small**:

- Optimizer: Adam
- Initial learning rate: 0.1 (Quello consigliato sarebbe 0.01)
- Nessun LR scheduler
- Batch size: 256
- $\alpha$ = 0.2 

NB: Ricordiamo che $\alpha$ è un coefficiente specifico per la funzione di loss che stiamo adottando, la $QUAD - QUAD$. 

Per quanto riguarda l'addestramento della rete **LSTM Large** abbiamo invece: 

- Optimizer: Adam
- Initial learning rate: 0.01
- LR Scheduler il quale divide per 10 il learning rate ogni 30 epoche
- Batch size: 256
- $\alpha$ = 0.4

Un **learning rate scheduler** in TensorFlow è una tecnica utilizzata per regolare automaticamente il tasso di apprendimento (learning rate) durante il processo di addestramento di una rete neurale. Il tasso di apprendimento è un parametro critico nella fase di ottimizzazione dei modelli di machine learning, poiché influenza quanto velocemente il modello impara dai dati di addestramento.
Nel nostro caso, avendo un elevato numero di epoche (100 per la LSTM large) il pericolo è quello dell'overfitting, modificare dinamicamente il learning rate diminuendolo ogni 30 epoche (in questo caso) ci permette di diminuire questo rischio. 

In particolare il nostro è fatto nella seguente maniera:

~~~ Python
def scheduler(epoch, lr):
    if epoch < 30:
        return lr
    elif epoch >= 30 and epoch < 60 :
        return 0.001
    elif epoch >= 60 and epoch < 90: 
        return 0.0001
    elif epoch >= 90: 
        return 0.00001
    else: 
        return 0.01; 
    

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose = 1)
~~~

I risultati ottenuti da questa rete sono stati: 
- loss: 127.6987 
- custom_score: 625.6381 
- val_loss: 125.9037 
- val_custom_score: 564.4070 
- lr: 1.0000e-05

In questo caso possiamo fare già dei paragoni con il lavoro svolto all'interno del paper poiché siamo riusciti a sistemare anche il funzionamento della funzione di S-score, la quale è stata implementata direttamente in fase di addestramento e mostra il suo risultato sotto il nome di *custom_score*. 
In particolare il *val_custom_score* non si discosta di molto rispetto a quello proposto dagli autori del paper che stiamo riproducendo, il quale si attesta sul valore di 446.89 per la variante non ottimizzata, come la nostra. 

Prossimi passi: 

Ottenere una versione definitiva della LSTM_small riaddestrando come fatto per LSTM_large. 