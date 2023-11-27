# Appunti Review su modelli di machine learning applicati a RUL

Per quanto riguarda il grande mondo del **PHM** esistono 3 diversi approcci: 

1. *Model-based*: 
   Sono difficili, costosi e richiedono grande conoscenza del mondo in questione (che non abbiamo)

2. ***Data driven***
   Sono la modalità a noi più congeniale, e soprattutto quella che possiamo sfruttare tramite gli algoritmi di Machine Learning su cui si basano per la maggior parte. 

3. *Ibridi*

Gli approcci ***Data Driven*** sono inoltre in grado di calcolare la RUL in <mark>maniera indiretta </mark> che cosa vuol dire però? 

<mark>Calcolare la RUL in maniera indiretta</mark>: vuol dire farlo basandosi sul degrado dei componenti del sistema in esame (*degradation based*)

In generale abbiamo 3 step da seguire:

1. Data acquisition (*ma non ci interessa*)

2. Costruzione *dell'indicatore di salute*

3. Predizione della RUL 

### Problemi dei metodi data driven

Per riuscire a rendere questo tipo di metodi efficaci è necessario però avere innanzitutto una grandissima quantità di dati su cui addestrare i modelli di ML e, soprattutto, c'è un trade-off in termini di precisione rispetto agli approcci model based

### Cosa troviamo in letteratura?

Dal documento che abbiamo esaminato risultano 15 dataset che si dividono essenzialmente in due macro gruppi: 

* Elementi posti in rotazione
  Per la maggiorparte provenienti dalla piattaforma PRONOSTIA (di cui possediamo un dataset completo)

* Turbofan engines
  Per la maggior parte provenienti dalla NASA, molto celebre il C-MAPSS dataset che viene estensivamente utilizzato all'interno di moltissime pubblicazioni e articoli 

Il problema di questi dataset risiede nel fatto che essi sono simulati e non riproducono situazioni operative reali. 

### Sfide per gli approcci *data-driven* basati su ML

- I modelli assumono che i dati utilizzati in fase di training e testing siano stati rilevati sotto condizioni simili (i.e appartengono alla stessa distribuzione)

- Notevoli difficoltà a relazionarsi col rumore 

- Complessità nella costruzione di un **indicatore di salute** valido 

### Perché utilizzare approcci di ML che siano diversi da quelli "Classici"?

Kalman Filter, Support Vector Machines ecc..., non considerano la rilevanza di serie temporali che riflettono micro-cambiamenti all'interno dello stato di salute del sistema. Di conseguenza è più indicato utilizzare algoritmi più complessi. In particolare vengono spesse tenute in considerazione le "Recurrent Neural Networks" (**RNN**) e delle loro derivazioni quali le *Long-short term memory* (**LTSM**). Le reti neurali permettono al modello di memorizzare *inferred pieces of knowledge* che possono essere acceduti e manipolati attraverso una memoria esterna. Un ulteriore vantaggio è la capacità di queste reti neurali di avere a che fare con complessità di tipo *non-lineare* addestrando reti neurali multi-livello. 

Tuttavia non ci sono solo aspetti positivi, alcunni problemi di questo tipo di algoritmi può essere l'elevato tempo di addestramento.

In particolare per quanto riguarda le **LTSM** esse non considerano dati multi-sensore e consumano risorse computazionali anche per quanto riguarda il deploy vero e proprio e non soltanto in fase di addestramento
