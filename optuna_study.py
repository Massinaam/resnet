import pickle
import optuna
import matplotlib.pyplot as plt

trials_data = [
    (0.2749, {'batch_size': 128, 'lr': 0.011992738123179698, 'optimizer': 'Adam', 'weight_decay': 0.00010947720438797557, 'step_size': 27, 'gamma': 0.4810556204792529, 'n_epochs': 17}),
    (0.5511, {'batch_size': 32, 'lr': 0.0016615034524852825, 'optimizer': 'Adam', 'weight_decay': 0.004956701444595925, 'step_size': 26, 'gamma': 0.3361110620801229, 'n_epochs': 13}),
    (0.7185, {'batch_size': 32, 'lr': 0.014457705482879614, 'optimizer': 'Adam', 'weight_decay': 0.0019716987611336154, 'step_size': 12, 'gamma': 0.8553633825541972, 'n_epochs': 25}),
    (0.1746, {'batch_size': 128, 'lr': 0.041210810525378054, 'optimizer': 'SGD', 'weight_decay': 1.0256975170135157e-05, 'step_size': 27, 'gamma': 0.2018793871353579, 'n_epochs': 11}),
    (0.5746, {'batch_size': 128, 'lr': 0.008927939334845326, 'optimizer': 'Adam', 'weight_decay': 0.0012513678959808377, 'step_size': 20, 'gamma': 0.7328351208788295, 'n_epochs': 14}),
    (0.5577, {'batch_size': 128, 'lr': 0.02746533543747569, 'optimizer': 'Adam', 'weight_decay': 0.00030844074152331887, 'step_size': 30, 'gamma': 0.3497058038653019, 'n_epochs': 26}),
    (0.3063, {'batch_size': 128, 'lr': 0.00074180614808309, 'optimizer': 'SGD', 'weight_decay': 0.001074164239368658, 'step_size': 24, 'gamma': 0.25854151390445934, 'n_epochs': 30}),
    (0.4, {'batch_size': 64, 'lr': 0.0003624169704824805, 'optimizer': 'SGD', 'weight_decay': 0.009264440152273972, 'step_size': 13, 'gamma': 0.1248207864773283, 'n_epochs': 16}),
    (0.6368, {'batch_size': 128, 'lr': 0.017720057273900653, 'optimizer': 'Adam', 'weight_decay': 0.009318540304575641, 'step_size': 16, 'gamma': 0.12083369396441856, 'n_epochs': 19}),
    (0.2076, {'batch_size': 64, 'lr': 0.021339731464102635, 'optimizer': 'SGD', 'weight_decay': 0.00036345991241247186, 'step_size': 30, 'gamma': 0.7919608772146843, 'n_epochs': 12}),
    (0.1419, {'batch_size': 32, 'lr': 0.09325464740600296, 'optimizer': 'SGD', 'weight_decay': 2.409778784133056e-06, 'step_size': 21, 'gamma': 0.5902881522927379, 'n_epochs': 10}),
    (0.1396, {'batch_size': 32, 'lr': 0.09557118987642071, 'optimizer': 'SGD', 'weight_decay': 2.6022126225668388e-06, 'step_size': 21, 'gamma': 0.6183009875295231, 'n_epochs': 10}),
    (0.1538, {'batch_size': 32, 'lr': 0.09959400527901933, 'optimizer': 'SGD', 'weight_decay': 2.1520243854213605e-06, 'step_size': 21, 'gamma': 0.6237434891931526, 'n_epochs': 10}),
    (0.4205, {'batch_size': 32, 'lr': 0.00010574086846793862, 'optimizer': 'SGD', 'weight_decay': 1.4704179154185823e-06, 'step_size': 19, 'gamma': 0.5785176136421983, 'n_epochs': 21}),
    (0.1506, {'batch_size': 32, 'lr': 0.05666853413879417, 'optimizer': 'SGD', 'weight_decay': 1.3003894765838174e-05, 'step_size': 17, 'gamma': 0.4820015653641816, 'n_epochs': 10}),
    (0.144, {'batch_size': 32, 'lr': 0.004451110987291546, 'optimizer': 'SGD', 'weight_decay': 9.580962348991401e-06, 'step_size': 23, 'gamma': 0.659584624232925, 'n_epochs': 15}),
    (0.0984, {'batch_size': 32, 'lr': 0.0789472738569447, 'optimizer': 'SGD', 'weight_decay': 3.968024253864851e-06, 'step_size': 17, 'gamma': 0.5454912096139305, 'n_epochs': 19}),
    (0.1104, {'batch_size': 32, 'lr': 0.005199524075076581, 'optimizer': 'SGD', 'weight_decay': 4.9569293225743016e-05, 'step_size': 15, 'gamma': 0.39668823745470333, 'n_epochs': 22}),
    (0.1348, {'batch_size': 64, 'lr': 0.004328427652631969, 'optimizer': 'SGD', 'weight_decay': 3.4474807786019887e-05, 'step_size': 10, 'gamma': 0.4257816724024485, 'n_epochs': 22}),
    (0.1349, {'batch_size': 32, 'lr': 0.0017848954055442055, 'optimizer': 'SGD', 'weight_decay': 6.217146184704268e-05, 'step_size': 15, 'gamma': 0.3826812137368674, 'n_epochs': 24}),
    (0.0966, {'batch_size': 32, 'lr': 0.007431466715022185, 'optimizer': 'SGD', 'weight_decay': 2.079773160121022e-05, 'step_size': 18, 'gamma': 0.5234796238150873, 'n_epochs': 19}),
    (0.1081, {'batch_size': 32, 'lr': 0.00717600035949742, 'optimizer': 'SGD', 'weight_decay': 3.0260394320275466e-05, 'step_size': 18, 'gamma': 0.5139621422080223, 'n_epochs': 19}),
    (0.1506, {'batch_size': 32, 'lr': 0.0019043984746506186, 'optimizer': 'SGD', 'weight_decay': 5.906340172106907e-06, 'step_size': 18, 'gamma': 0.5413544425526688, 'n_epochs': 18}),
    (0.1071, {'batch_size': 32, 'lr': 0.00821693436250772, 'optimizer': 'SGD', 'weight_decay': 2.2863273729587802e-05, 'step_size': 18, 'gamma': 0.725223031888795, 'n_epochs': 20}),
    (0.0980, {'batch_size': 32, 'lr': 0.03595361253277185, 'optimizer': 'SGD', 'weight_decay': 2.0041326436171215e-05, 'step_size': 14, 'gamma': 0.7097610771420745, 'n_epochs': 20}),
    (0.1227, {'batch_size': 64, 'lr': 0.03606570221310026, 'optimizer': 'SGD', 'weight_decay': 5.170000392246285e-06, 'step_size': 13, 'gamma': 0.6904176108050969, 'n_epochs': 23}),
    (0.1093, {'batch_size': 32, 'lr': 0.05564795060191872, 'optimizer': 'SGD', 'weight_decay': 0.00015037933336921365, 'step_size': 10, 'gamma': 0.8765986352958959, 'n_epochs': 17}),
    (0.0872, {'batch_size': 32, 'lr': 0.028114831155793277, 'optimizer': 'SGD', 'weight_decay': 1.639675366866713e-05, 'step_size': 15, 'gamma': 0.44721303680779584, 'n_epochs': 27}),
    (0.0903, {'batch_size': 32, 'lr': 0.02723443894006373, 'optimizer': 'SGD', 'weight_decay': 1.5780903232097898e-05, 'step_size': 14, 'gamma': 0.44517489206220884, 'n_epochs': 28}),
    (0.2212, {'batch_size': 64, 'lr': 0.012715291243662582, 'optimizer': 'Adam', 'weight_decay': 0.0001076144760001516, 'step_size': 12, 'gamma': 0.45734100597742505, 'n_epochs': 28}),
    (0.1184, {'batch_size': 32, 'lr': 0.002705578433071044, 'optimizer': 'SGD', 'weight_decay': 1.4794675160356895e-05, 'step_size': 15, 'gamma': 0.2747847522085727, 'n_epochs': 27}),
    (0.0780, {'batch_size': 32, 'lr': 0.027777915561293935, 'optimizer': 'SGD', 'weight_decay': 2.151610178706004e-05, 'step_size': 14, 'gamma': 0.45484462903641865, 'n_epochs': 30}),
    (0.0846, {'batch_size': 32, 'lr': 0.022161562153468477, 'optimizer': 'SGD', 'weight_decay': 6.790670064144368e-05, 'step_size': 12, 'gamma': 0.4591379488758971, 'n_epochs': 30}),
    (0.0828, {'batch_size': 32, 'lr': 0.02394017397266421, 'optimizer': 'SGD', 'weight_decay': 7.0524274274823e-05, 'step_size': 11, 'gamma': 0.450789696138724, 'n_epochs': 30}),
    (0.3147, {'batch_size': 32, 'lr': 0.016066352976427636, 'optimizer': 'Adam', 'weight_decay': 0.00022061972159259383, 'step_size': 11, 'gamma': 0.36780546448612755, 'n_epochs': 30}),
    (0.0804, {'batch_size': 32, 'lr': 0.04858349155309898, 'optimizer': 'SGD', 'weight_decay': 6.693445758548756e-05, 'step_size': 12, 'gamma': 0.3041953243276668, 'n_epochs': 29}),
    (0.4404, {'batch_size': 32, 'lr': 0.05854628825939452, 'optimizer': 'Adam', 'weight_decay': 7.208263271964911e-05, 'step_size': 12, 'gamma': 0.3076840320325816, 'n_epochs': 29}),
    (0.1361, {'batch_size': 128, 'lr': 0.010920722320503225, 'optimizer': 'SGD', 'weight_decay': 0.00012341028038658134, 'step_size': 11, 'gamma': 0.188571108028696, 'n_epochs': 30}),
    (0.0809, {'batch_size': 32, 'lr': 0.04091712610134188, 'optimizer': 'SGD', 'weight_decay': 0.0004027744792323627, 'step_size': 11, 'gamma': 0.31641862707741336, 'n_epochs': 25}),
    (0.5013, {'batch_size': 128, 'lr': 0.04854509311607941, 'optimizer': 'Adam', 'weight_decay': 0.0004625769093583895, 'step_size': 10, 'gamma': 0.3240349694581185, 'n_epochs': 25}),
    (0.2081, {'batch_size': 32, 'lr': 0.0006341834453378504, 'optimizer': 'SGD', 'weight_decay': 0.0006177075780318213, 'step_size': 13, 'gamma': 0.2255435405539195, 'n_epochs': 26}),
    (0.0821, {'batch_size': 32, 'lr': 0.0229446325106797, 'optimizer': 'SGD', 'weight_decay': 0.00019799745762813362, 'step_size': 11, 'gamma': 0.2909732933496185, 'n_epochs': 29}),
    (0.0682, {'batch_size': 32, 'lr': 0.0358408115435323, 'optimizer': 'SGD', 'weight_decay': 0.002345625688196673, 'step_size': 12, 'gamma': 0.28261595872305967, 'n_epochs': 28}),
    (0.0683, {'batch_size': 32, 'lr': 0.03866688726191991, 'optimizer': 'SGD', 'weight_decay': 0.002840016614857349, 'step_size': 13, 'gamma': 0.16533000008401338, 'n_epochs': 28}),
    (0.1059, {'batch_size': 32, 'lr': 0.039034812241647995, 'optimizer': 'SGD', 'weight_decay': 0.0032036115025367348, 'step_size': 14, 'gamma': 0.17240094739056644, 'n_epochs': 28}),
    (0.0878, {'batch_size': 128, 'lr': 0.06757204120255368, 'optimizer': 'SGD', 'weight_decay': 0.0018191549341386956, 'step_size': 13, 'gamma': 0.24881181927742327, 'n_epochs': 26}),
    (0.1669, {'batch_size': 64, 'lr': 0.016386905508285577, 'optimizer': 'SGD', 'weight_decay': 0.0052357347980452424, 'step_size': 28, 'gamma': 0.14297794443692807, 'n_epochs': 27}),
    (0.8048, {'batch_size': 32, 'lr': 0.04220365842448873, 'optimizer': 'Adam', 'weight_decay': 0.0008658690261659421, 'step_size': 16, 'gamma': 0.2316254389499045, 'n_epochs': 25}),
    (0.0685, {'batch_size': 32, 'lr': 0.0314892817245799, 'optimizer': 'SGD', 'weight_decay': 0.0028710980585713772, 'step_size': 13, 'gamma': 0.10439747389625614, 'n_epochs': 29}),
    (0.1025, {'batch_size': 32, 'lr': 0.07232216239473974, 'optimizer': 'SGD', 'weight_decay': 0.003519681409074442, 'step_size': 16, 'gamma': 0.11048078103465978, 'n_epochs': 29})
]

study = optuna.create_study(direction="minimize", study_name="reconstructed_study")

for value, params in trials_data:
    trial = study.ask(fixed_distributions={k: optuna.distributions.FloatDistribution(0, 1) for k in params})
    trial.params.update(params)
    study.tell(trial, value)

with open("reconstructed_study.pkl", "wb") as f:
    pickle.dump(study, f)

# Load the study from the saved pickle file
with open("reconstructed_study.pkl", "rb") as f:
    study = pickle.load(f)
#%%
# Plot the optimization progress (objective value over trials)
optuna.visualization.plot_optimization_history(study)
plt.title('Optimization Progress')
plt.savefig("opti_prog.png")
plt.show()

# Plot hyperparameter importance
optuna.visualization.plot_param_importances(study)
plt.title('Hyperparameter Importance')
plt.savefig("hyper_imp.png")
plt.show()

# Plot distribution of objective values
optuna.visualization.plot_parallel_coordinate(study)
plt.title('Parallel Coordinate Plot for Objective Value Distribution')
plt.savefig("parall.png")
plt.show()

# Plot batch_size vs objective value
optuna.visualization.plot_contour(study, params=['batch_size', 'lr'])
plt.title('Batch Size vs Objective Value')
plt.savefig("batch.png")
plt.show()
