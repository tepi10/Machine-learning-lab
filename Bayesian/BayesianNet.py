import matplotlib.pyplot as plt
import networkx as nx

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


# 1️⃣ Define Network Structure
model = DiscreteBayesianNetwork([
    ('Age', 'HeartDisease'),
    ('Cholesterol', 'HeartDisease'),
    ('BloodPressure', 'HeartDisease'),
    ('Smoking', 'HeartDisease')
])


# 2️⃣ Define CPTs
cpd_age = TabularCPD('Age', 2, [[0.6], [0.4]])

cpd_chol = TabularCPD('Cholesterol', 2, [[0.7], [0.3]])

cpd_bp = TabularCPD('BloodPressure', 2, [[0.65], [0.35]])

cpd_smoke = TabularCPD('Smoking', 2, [[0.8], [0.2]])


cpd_hd = TabularCPD(
    variable='HeartDisease',
    variable_card=2,
    values=[
        [0.9,0.8,0.7,0.6,0.7,0.6,0.5,0.4,
         0.8,0.7,0.6,0.5,0.6,0.5,0.4,0.3],
        [0.1,0.2,0.3,0.4,0.3,0.4,0.5,0.6,
         0.2,0.3,0.4,0.5,0.4,0.5,0.6,0.7]
    ],
    evidence=['Age','Cholesterol','BloodPressure','Smoking'],
    evidence_card=[2,2,2,2]
)


# 3️⃣ Add CPTs
model.add_cpds(cpd_age, cpd_chol, cpd_bp, cpd_smoke, cpd_hd)

print("Model Valid:", model.check_model())


# 4️⃣ Inference
infer = VariableElimination(model)

result = infer.query(
    variables=['HeartDisease'],
    evidence={'Age':1, 'Smoking':1}
)

print("\nHeart Disease Probability:")
print(result)


# 5️⃣ Graph Visualization
G = nx.DiGraph()

G.add_edges_from([
    ('Age','HeartDisease'),
    ('Cholesterol','HeartDisease'),
    ('BloodPressure','HeartDisease'),
    ('Smoking','HeartDisease')
])

pos = nx.spring_layout(G)

plt.figure(figsize=(7,5))
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', arrows=True)

plt.title("Bayesian Network Graph for Heart Disease Prediction")
plt.show()
