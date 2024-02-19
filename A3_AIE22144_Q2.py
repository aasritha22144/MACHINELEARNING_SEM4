import pandas as pd
import numpy as np

#Taking dataset from excel sheet 
file = pd.read_excel(r"C:\Users\aasri\Downloads\Lab Session1 Data.xlsx",sheet_name="Purchase data")
S = file[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].to_numpy()
C = file['Payment (Rs)'].to_numpy().reshape(-1, 1)
print(f"A = {S}")
print(f"C = {C}")

#Finding the pseudo inverse
pseudo_inverse = np.linalg.pinv(S)

model_vector_X = np.dot(pseudo_inverse,C)
print(f"The model vector X for predicting the cost of the products available with the vendor = {model_vector_X.flatten()}")

