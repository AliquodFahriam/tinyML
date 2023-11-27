# BOA-MKRVM bearing reliability

Il contenuto di questo documento contiene degli spunti interessanti. Per quanto la mia volontà rimanga sempre quella di voler utilizzare una rete neurale ricorrente (**RNN**) di tipo **LSTM**, come suggerito dalla maggior parte dello stato dell'arte, gli strumenti utilizzati per l'ottimizzazione degli iperparametri del modello utilizzato in questo documento e quelli utilizzati per l'intero processo di *denoising* sono di particolare interesse dati i risultati ottenuti dagli autori.

## Architettura del progetto

![](/home/aliquodfahriam/ArchHome/.config/marktext/images/2023-07-20-11-58-33-image.png)

Per quanto riguarda le tecnologie utilizzate rimando al secondo file *.md* che si trova all'interno di questa cartella. 

Per quanto riguarda i risultati della fase di denoising abbiamo una differenza incredibile tra il pre e il post: 

![](/home/aliquodfahriam/ArchHome/.config/marktext/images/2023-07-20-12-00-02-image.png)

![](/home/aliquodfahriam/ArchHome/.config/marktext/images/2023-07-20-12-00-24-image.png)

Un ulteriore approccio interessante è dato dalla volontà degli autori di non fermarsi mai a decidere manualmente gli "*iperparametri*" dei vari algoritmi, ma di utilizzare sempre algoritmi di ricerca o di ottimizzazione. 

In particolare per quanto riguarda gli iperparametri dell'algoritmo MKRVM viene utilizzato il ***Bayesian Optimization Algorithm*** (Il cui funzionamento è spiegato nel documento di cui sopra), mentre per quanto riguarda **AMCKD**, ovvero il secondo "blocco" addetto al denoising, si utilizza un algoritmo di ricerca detto ***Sparrow Search*** (che possiamo trovare all'interno dei riferimenti dell'articolo). 
