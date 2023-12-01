# A primer for tinyML predictive Maintenance: Input and Model Optimisation

##### Predictive maintenance vs Reactive maintenance vs preventive maintenance:

Il problema degli ultimi due approcci è che nel primo caso possono essere causati gravi danni da fallimenti non affrontati prima della loro occorrenza, nel secondo caso, invece, potrebbero essere sostituiti dei pezzi perfettamente funzionanti portando ad un notevole spreco. 

Per questo la prospettiva del *predictive maintenance* risulta più allettante per noi. 

Esistono tre approcci generali: 

- Knowledge based: Si basa sulla conoscenza del campo in cui si va a lavorare, il degrado di un componente viene modellizzato e il modello poi applicato per predirre la durata della vita del componente 

- Machine Learning 

- Deep Learning

##### Perché TinyML:

La maggior parte dei sistemi di Predictive Maintenance sono al momento sul cloud, questo può portare a numerosi problemi di varia natura quali: 

- Sicurezza dei dati 

- Latenza legata alla rete 

- Affidabilità del sistema 

- Consumo elevato di energia

Lo sviluppo su microcontrollore risolve questi problemi presentando il conto per quanto riguarda le capacità computazionali. 

##### Il problema dei dati:

TinyML è ancora agli albori, di conseguenza è difficile recuperare dei dati che siano sufficientemente adatti per lo sviluppo di algoritmi di ML per PdM. La maggior parte dei dati a nostra disposizione sono *sbilanciati* oppure non etichettati, il che presenta un grande problema in fase di addestramento. 

Per *dataset sbilanciati* intendiamo il fatto che (per natura stessa del problema) un dataset contiene molti più campioni di funzionamento "normale" che campioni di funzionamento anomalo o fallimentare. 

La letteratura propone di utilizzare un approccio basato su modelli generativi o transfer learning per risolvere questo problema. <a href='https://arxiv.org/abs/1912.07383'>Link</a>

##### Librerie e Software:

- TFLM (*Tensorflow Lite Micro*): composta da un interprete e un convertitore, non permette di effettuare la fase di addestramento (delegata a *Tensorflow*)

- Microsoft Embedded Learning Library

- ARM-NN

- Sklearnporter

- $\mu$Tensor

La maggior parte di queste tuttavia sono adatte per approcci di ML ma non di Deep Learning. 

##### Hardware:

 La maggior parte dei microcontrollori su cui possiamo sviluppare sono basati su chip ARM Cortex, i quali a loro volta si dividono in due categorie: 

- Cortex-M (Nucleo, Arduino, ecc...): è il processore con la maggior efficienza energetica, tuttavia ha scarse performance. 

- Cortex-A (Raspberry Pi): è più energivoro, tuttavia ha prestazioni migliori.  

(Viene anche menzionata la piattaforma PULP, la quale dovrebbe migliorare la fase di inferenza <a href = 'https://arxiv.org/abs/1908.11263'>Link</a> )

##### Datasets:

- CMAPSS

- ToyADMOS: contiene registrazioni audio di giocattoli in condizioni di operatività normale e anomala 

- MIMII: Contiene registrazioni audio di macchine industriali in condizioni normali e anomale 

##### Tecniche di ottimizzazione

- Quantizzazione 

- Pruning

- Clustering 

- Knowledge Distillation

- Rimozione delle operazioni 

- Architettura a cascata

Altri riferimenti utili: 

David, R., Duke, J., Jain, A., Janapa Reddi, V., Jeffries, N., Li, J., Kreeger, N.,
Nappier, I., Natraj, M., et al.: Tensorflow lite micro: Embedded machine learning
for tinyml systems. Proc. Machine Learning and Systems 3, 800–811 (2021)

Banbury, C., Reddi, V.J., Torelli, P., Jeffries, N., Kiraly, C., et al.: MLPerf Tiny
Benchmark. In: Thirty-fifth Conference on Neural Information Processing Systems
Datasets and Benchmarks Track (Round 1) (2021)

Hinton, G., Vinyals, O., Dean, J., et al.: Distilling the knowledge in a neural net-
work. arXiv preprint arXiv:1503.02531 2(7) (2015)
