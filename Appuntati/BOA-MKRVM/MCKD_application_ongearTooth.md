# Maximum correlated Kurtosis deconvolution and application on gear tooth chip fault detection



### DISCLAIMER: Quello che viene detto in questo documento per quanto riguarda lo stato dell'arte ormai rappresenta informazioni datate le quali non risultano più rappresentative della condizione odierna di quest'ultimo.



Questo metodo di convoluzione viene presentato all'interno del documento con l'obiettivo di individuare *faults* di ingranaggi e cuscinetti a sfera. 

*I dati sperimentali vengono da un gearbox con problemi di denti mancanti e i risultati sono comparati tra vibrazioni <mark>healthy</mark> e <mark> faulty</mark>*. 

Il nostro obiettivo per quanto riguarda questo tipo di deconvoluzione è quello di utilizzarla nella riduzione del rumore per quanto riguarda il segnale proveniente da accelerometri posti sul corpo rotante. In sostanza come fatto per <a href ="./BOA-MKRVM_bearing_reliability.md"> il documento precedente </a>. 

Precedentemente spesso si utilizzavano approcci basati su modelli "*autoregressive*" in combinazione con *Minimum Entropy Deconvolution*, anche chiamati metodi **ARMED**.

La tecnica MED punrna ad estrarre gli impulsi di fault minimizzando allo stesso tempo il rumore riuscendo a ottenere buoni risultati anche quando questo è molto elevato. 
Tuttavia questo tipo di tecnica non è in grado di mettere in luce la periodicità degi *fault impluses*. 



Per comprendere come viene implementato l'algoritmo **MKCD** si rimanda al codice matlab nel file *.zip* da cui potrebbe essere semplice ricavare una versione in python  

 
