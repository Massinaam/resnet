import pickle
import optuna
import matplotlib.pyplot as plt
import numpy as np

# Charger l'étude à partir du fichier pickle
with open("reconstructed_study.pkl", "rb") as f:
    study = pickle.load(f)

# Vérification de la bonne charge de l'étude
if study is None:
    print("L'étude n'a pas été chargée correctement.")
else:
    print("L'étude a été chargée avec succès.")

# 1. Plot de l'évolution de la valeur de l'objectif au fil des trials
objective_values = [trial.value for trial in study.trials]
plt.figure(figsize=(10, 6))
plt.plot(range(len(objective_values)), objective_values, marker='o')
plt.title("Optimization Progress")
plt.xlabel("Trial Number")
plt.ylabel("Objective Value")
plt.savefig("opti_prog.png")
plt.show()

# 2. Plot des importances des hyperparamètres (en utilisant un calcul manuel)
param_names = [param for param in study.best_trial.params]
param_values = [study.best_trial.params[param] for param in param_names]

plt.figure(figsize=(10, 6))
plt.barh(param_names, param_values)
plt.title("Hyperparameter Importance")
plt.xlabel("Hyperparameter Value")
plt.ylabel("Parameter")
plt.savefig("hyper_imp.png")
plt.show()

# 3. Plot des coordonnées parallèles pour la distribution des valeurs de l'objectif
# Nous utilisons ici un calcul direct des corrélations des paramètres avec la valeur objective
param_values = [trial.params['batch_size'] for trial in study.trials]
objective_values = [trial.value for trial in study.trials]

plt.figure(figsize=(10, 6))
plt.scatter(param_values, objective_values)
plt.title("Parallel Coordinate Plot for Objective Value Distribution")
plt.xlabel("Batch Size")
plt.ylabel("Objective Value")
plt.savefig("parall.png")
plt.show()

# 4. Plot contour pour batch_size vs objective value
batch_sizes = [trial.params['batch_size'] for trial in study.trials]
learning_rates = [trial.params['lr'] for trial in study.trials]

# Créer un maillage pour contour
plt.figure(figsize=(10, 6))
plt.scatter(batch_sizes, learning_rates, c=objective_values, cmap='viridis')
plt.title("Batch Size vs Objective Value")
plt.xlabel("Batch Size")
plt.ylabel("Learning Rate")
plt.colorbar(label='Objective Value')
plt.savefig("batch.png")
plt.show()
