import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import CategoricalEncoder

train = pd.read_csv('train.csv')

# Cabin allocations https://www.encyclopedia-titanica.org/cabins.html
# label = LabelBinarizer()
# hot = OneHotEncoder(handle_unknown = 'ignore', n_values=['male', 'female'])
categorical = CategoricalEncoder()
pipe = Pipeline([('Categorical Encoder', categorical)])

# pipe = Pipeline([('Label Binarizer', label)])

# train['Sex'] = train['Sex'].replace({'male': 0, 'female': 1})
# train.head()
# train.describe()

train = train[['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Age', 'Survived']]

pipe.fit(train)

imputer = Imputer()


train.describe()

vaiq = train.drop('Survived', axis=1)

vaiq = pd.DataFrame(imputer.fit_transform(vaiq),
                    index=vaiq.index, columns=vaiq.columns)

vaiq.describe()
vaiq.head(3)

x_train, x_test, y_train, y_test = train_test_split(vaiq,
                                                    train['Survived'],
                                                    test_size=0.3,
                                                    random_state=42)

arvore = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
arvore.fit(x_train, y_train)

export_graphviz(arvore, out_file='arvore_titanic.dot',
                label=['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Age'])
get_ipython().system('dot -Tpng arvore_titanic.dot -o arvore_titanic.png')

previsao = arvore.predict(x_test)
(previsao == y_test).sum()/(len(y_test))
len(y_test)
f1_score(y_test, previsao)

forest = RandomForestClassifier(n_estimators=30, max_depth=3,
                                random_state=42, max_features=2)
floresta = forest.fit(x_train, y_train)
floresta
prevendo = floresta.predict(x_test)
(prevendo == y_test).sum()/(len(y_test))
f1_score(y_test, prevendo)
floresta_completa = forest.fit(vaiq, train['Survived'])
test_kaggle = pd.read_csv('test.csv')
passenger_id = test_kaggle['PassengerId']
test_kaggle = test_kaggle[['Pclass', 'Sex',
                           'SibSp', 'Parch',
                           'Fare', 'Age']].replace({'male': 0, 'female': 1})
test_kaggle = pd.DataFrame(imputer.transform(test_kaggle),
                           index=test_kaggle.index,
                           columns=test_kaggle.columns)
envio = floresta_completa.predict(test_kaggle)
submission = pd.DataFrame({'PassengerId': passenger_id, 'Survived': envio})
submission.index = submission['PassengerId']
submission = submission.drop('PassengerId', axis=1)
submission.head()
submission.to_csv('sub.csv')
