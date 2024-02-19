import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

#Taking dataset from excel sheet 
file = pd.read_excel(r"C:\Users\aasri\Downloads\Lab Session1 Data.xlsx", sheet_name="Purchase data")

file['Customer'] = file['Payment (Rs)'].apply(lambda x: 'RICH' if x > 200 else 'POOR')
X = file[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']]
Y = file['Customer'] 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=42)
 #Classification of KNN
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, Y_train)

prediction = classifier.predict(X_test)
print("Classification Report")
print(classification_report(Y_test, prediction, zero_division=1))
