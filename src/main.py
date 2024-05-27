import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

gender = "C:\\Ludmila\\Projetinhos\\titanicKaggle\\titanicData\\gender_submission.csv"
test = "C:\\Ludmila\\Projetinhos\\titanicKaggle\\titanicData\\test.csv"
train="C:\\Ludmila\\Projetinhos\\titanicKaggle\\titanicData\\train.csv"

titanicData = pd.read_csv(train)
titanicData.info()
titanicData.describe()
titanicData.describe().columns

dfNum = titanicData[['Age', 'SibSp', 'Parch', 'Fare']]
dfNum

dfCat =titanicData[['Survived', 'Pclass', 'Sex', 'Ticket', 'Cabin', 'Embarked']]
dfCat

for i in dfNum.columns:
    plt.hist(dfNum[i])
    plt.title(i)
    plt.show()

print(dfNum.corr())
sns.heatmap(dfNum.corr())

pd.pivot_table(titanicData, index= "Survived", values= ["Age", "SibSp", "Parch", "Fare"])

for i in dfCat.columns:
    ax = sns.barplot(x=dfCat[i].value_counts().index, y=dfCat[i].value_counts())
    ax.set_title(i)
    plt.show()

print(pd.pivot_table(titanicData, index= 'Survived', columns='Pclass', values='Ticket', aggfunc='count'))
print(pd.pivot_table(titanicData, index= 'Survived', columns='Sex', values='Ticket', aggfunc='count'))
print(pd.pivot_table(titanicData, index= 'Survived', columns='Embarked', values='Ticket', aggfunc='count'))

titanicData['cabin_multiple'] = titanicData['Cabin'].apply(lambda x: 0 if pd.isna(x) else len(x.split(" ")))
print(titanicData[['Cabin', 'cabin_multiple']])
titanicData['cabin_multiple'].value_counts()
print(pd.pivot_table(titanicData, index= 'Survived', columns='cabin_multiple', values='Ticket', aggfunc='count'))

titanicData['cabin_adv'] = titanicData['Cabin'].apply(lambda x: str(x)[0])
print(titanicData[['Cabin', 'cabin_adv']])
titanicData['cabin_adv'].value_counts()
print(pd.pivot_table(titanicData, index= 'Survived', columns='cabin_adv', values='Name', aggfunc='count'))


titanicData['numeric_ticket'] = titanicData.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
titanicData['ticket_letters'] = titanicData.Ticket.apply(lambda x: " ".join(x.split(" ")[:-1]).replace('.', ' ').replace('/', '').lower() if len(x.split(' ')[:-1])>0 else 0)
titanicData['numeric_ticket'].value_counts()
titanicData['ticket_letters'].value_counts()

print(pd.pivot_table(titanicData, index= 'Survived', columns='ticket_letters', values='Ticket', aggfunc='count'))
print(pd.pivot_table(titanicData, index= 'Survived', columns='numeric_ticket', values='Ticket', aggfunc='count'))

titanicData.Name.head(50)
titanicData['name_title'] = titanicData.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())
titanicData['name_title'].value_counts()



titanicData['cabin_multiple'] = titanicData.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
titanicData['cabin_adv'] = titanicData.Cabin.apply(lambda x: str(x)[0] if pd.notna(x) else 'Unknown')
titanicData['numeric_ticket'] = titanicData.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
titanicData['ticket_letters'] = titanicData.Ticket.apply(
    lambda x: " ".join(x.split(' ')[:-1]).replace('.', '').replace('/', '').lower() if len(x.split(' ')[:-1]) > 0 else 'None'
)
titanicData['name_title'] = titanicData.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())

titanicData['Age'] = titanicData.Age.fillna(titanicData.Age.mean())
titanicData['Fare'] = titanicData.Fare.fillna(titanicData.Fare.mean())
titanicData.dropna(subset=['Embarked'], inplace=True)

titanicData['norm_sibsp'] = np.log(titanicData.SibSp + 1)
titanicData['norm_sibsp'].hist()
plt.title('Normalized SibSp')
plt.show()

titanicData['norm_fare'] = np.log(titanicData.Fare + 1)
titanicData['norm_fare'].hist()
plt.title('Normalized Fare')
plt.show()

titanicData['Pclass'] = titanicData.Pclass.astype(str)






