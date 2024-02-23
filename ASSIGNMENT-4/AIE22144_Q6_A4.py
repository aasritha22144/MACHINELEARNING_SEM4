import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
df = pd.read_csv(r'Food Ingredients and Recipe Dataset with Image Name Mapping.csv')
print("Column names in the dataset:", df.columns)
y_column_name = 'Instructions'  
X = df['Cleaned_Ingredients']
y = df[y_column_name]
df.dropna(subset=['Cleaned_Ingredients', 'Instructions'], inplace=True)
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(df['Cleaned_Ingredients'])
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, df['Instructions'], test_size=0.3, random_state=42
)
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred)
print("Accuracy on the test set:", accuracy_test)