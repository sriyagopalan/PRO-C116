import pandas as pd;
import plotly.express as px;
from sklearn.linear_model import LogisticRegression;
from sklearn.metrics import accuracy_score;
import numpy as np;

data = pd.read_csv('data.csv');

score = data['TOEFL Score'].tolist();
chance = data['Chance of admit'].tolist();

fig = px.scatter(data, x = score, y = chance);
#fig.show();

m, c = np.polyfit(score, chance, 1);

Y = [];
for x in score:
    y_value = m * x + c;
    if y_value < 0.5:
        Y.append(0);
    else:
        Y.append(1);

accuracy = accuracy_score(chance, Y);
print(accuracy);