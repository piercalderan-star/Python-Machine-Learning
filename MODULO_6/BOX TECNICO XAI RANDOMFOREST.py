'''
BOX TECNICO – Explainable AI (XAI)
Cos’è la XAI?

La Explainable AI (XAI) è l’insieme di tecniche, metodi e strumenti che
permettono di comprendere e visualizzare il comportamento di un modello di Machine Learning,
soprattutto quando questo è complesso o considerato
una black-box (es. Random Forest, Gradient Boosting).

La XAI risponde a domande come:
Perché il modello ha fatto questa predizione?
Quali feature influenzano maggiormente l’output?
Come cambia la predizione se modifico una variabile?
Il modello si comporta in modo coerente e affidabile?

Interpretabilità Globale vs Locale
Interpretabilità Globale

Analizza l’intero modello, mostrando le tendenze generali.
Esempi:
Feature Importance
misura quanto ogni variabile contribuisce, in media, al modello.

Permutation Importance
valuta il calo di performance quando una feature viene “shuffle-ata”.

Partial Dependence Plot (PDP)
mostra come in media cambia la predizione al variare di una sola (o due) feature.

Interpretabilità Locale
Spiega una singola predizione.

Esempi:
LIME
genera un modello locale, semplice e interpretabile (lineare o alberello),
che approssima il modello complesso in una zona ristretta
vicino all’istanza da spiegare.

SHAP
mostra quanto ogni feature “spinge” la predizione verso l’alto
o verso il basso, sando i concetti della teoria dei giochi (Shapley Values).

Quando usare la XAI?
Per modellistica regolamentata (finanza, assicurazioni, sanità).
Per motivare decisioni a manager e clienti.
Per rilevare bias, anomalie o comportamenti inaspettati.
Per costruire modelli più robusti, verificando l’importanza reale delle feature.
Per aumentare la fiducia nel modello in applicazioni critiche.

Strumenti più usati
Scikit-Learn
Feature importance
Permutation importance
Partial Dependence Plot

SHAP
Per modelli Tree (RF, XGBoost, LightGBM)
Per modelli NN (DeepExplainer)

LIME
Per qualsiasi modello (tabellare, immagini, testo)
Best Practices
Non affidarsi solo alla feature importance → usare anche permutation importance.
Combinare tecnica globale (PDP) e locale (SHAP/LIME).
Scegliere XAI in base al contesto:
SHAP → modelli ad albero
LIME → modelli qualsiasi
PDP → comprensione generale
Effettuare XAI sia su train sia su test: differenze grandi = rischio overfitting.

BOX TECNICO – Random Forest
Cos’è la Random Forest?

La Random Forest è un algoritmo di Machine Learning basato su ensemble learning,
costruito come insieme di molti alberi decisionali (Decision Tree) addestrati in parallelo.
Ogni albero prende decisioni diverse grazie a:

Bootstrap sampling → ogni albero vede un sottoinsieme casuale del dataset;

Random feature selection → a ogni split, l’albero può usare solo un sottoinsieme casuale delle feature.

La predizione finale è la combinazione degli alberi:

Regressione → media dei valori previsti

Classificazione → voto di maggioranza

Perché funziona così bene?
La Random Forest risolve i limiti dei singoli alberi decisionali:

Un singolo albero può "overfittare" (overfitting=sovradattamento)
Una foresta di alberi con diversificazione → riduce la varianza, aumenta la robustezza

La logica è simile al principio statistico:
“Molti modelli deboli + casualità = un modello forte”

Parametri chiave

n_estimators
numero di alberi → più alberi = migliore stabilità (tipico: 100–500)

max_depth
profondità massima degli alberi

più grande → più complesso e preciso, ma rischio overfitting

None → albero completamente espanso

max_features
quante feature considerare a ogni split

“sqrt” (default in classificazione)

“log2”

numero fisso (es. 5)

min_samples_split / min_samples_leaf
controllano la complessità degli alberi

Vantaggi

Funziona molto bene out-of-the-box
Robusto al rumore
Gestisce bene feature numeriche e categoriche (dopo encoding)
Fornisce feature importance nativa
Facile da usare e difficile da far “impazzire”

Svantaggi

Poco interpretabile senza XAI → soluzione: SHAP / Permutation Importance
Modello pesante:
molti alberi = più RAM
predizione più lenta rispetto a un singolo albero
Non ideale per dati ad alta dimensionalità (10.000+ feature) dove i modelli lineari possono funzionare meglio.

Quando usare Random Forest?
Dati tabellari (numerici + categorici) → perfetta
Dataset non troppo grandi (10k–1M righe)
Situazioni dove serve robustezza e performance senza tuning pesante

Per problemi di:
previsione prezzi (case, prodotti…)
credit scoring
classificazione generale

Best Practices
Usare almeno 200–500 alberi per stabilità
Limitare max_depth se il dataset è rumoroso
Usare class_weight='balanced' in caso di classi sbilanciate
Non fidarsi solo della feature importance!
→ usare permutation importance e SHAP per capire davvero cosa succede
'''
