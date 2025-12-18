import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

df = pd.read_csv("prezzi_case_ml.csv")

X = df[["superficie", "stanze", "zona"]]
y = df["prezzo"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

print("Score:", model.score(X_test, y_test))

# Feature importance
importances = model.feature_importances_
feature_names = X.columns

for name, imp in zip(feature_names, importances):
    print(f"{name}: {imp:.3f}")

# Plot 1
plt.bar(feature_names, importances)
plt.title("Feature Importance (RandomForest)")
plt.ylabel("Importanza")
plt.show()

#xai_permutation_importance.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

df = pd.read_csv("prezzi_case_ml.csv")

X = df[["superficie", "stanze", "zona"]]
y = df["prezzo"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

r = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42)

for i in r.importances_mean.argsort()[::-1]:
    i = int(i)
    print(f"{X.columns[i]}: {r.importances_mean[i]:.3f} ± {r.importances_std[i]:.3f}")

# Plot 2
sorted_idx = r.importances_mean.argsort()
plt.boxplot(r.importances[sorted_idx].T, vert=False, tick_labels=X.columns[sorted_idx])
plt.title("Permutation Importance (test set)")
plt.xlabel("Decrease in score")
plt.show()



#xai_partial_dependence.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

df = pd.read_csv("prezzi_case_ml_float.csv")

X = df[["superficie", "stanze", "zona"]]
y = df["prezzo"]
print("X",X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

print(X_train, y_train)

features = ["superficie", "stanze"]

PartialDependenceDisplay.from_estimator(
    model,
    X_train,
    features=features,
    kind="average"
)

#plot 3
plt.suptitle("Partial Dependence Plot")
plt.show()

#xai_shap_rf.py

import shap
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Es: classificazione Titanic
df = pd.read_csv("titanic_ml.csv")

X = df[["età", "classe", "tariffa"]]
y = df["sopravvissuto"]
#print(X,y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
print("Accuracy:", model.score(X_test, y_test))

# ---- SHAP ----
'''
SHAP (SHapley Additive ExPlanations) è un approccio basato
sulla teoria dei giochi per spiegare l'output di qualsiasi modello
di apprendimento automatico.
Collega l'allocazione ottimale dei crediti
con spiegazioni locali utilizzando i classici
valori di Shapley della teoria dei giochi
e le relative estensioni 
'''

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Gestiamo sia il caso "lista di array" che il caso "array singolo"
if isinstance(shap_values, list):
    # Classificazione binaria: prendiamo i valori della classe positiva (in genere indice 1)
    shap_values_to_plot = shap_values[1]
else:
    # Già un array (n_samples, n_features)
    shap_values_to_plot = shap_values
#plot 4
shap.summary_plot(shap_values_to_plot, X_test, feature_names=X.columns)

