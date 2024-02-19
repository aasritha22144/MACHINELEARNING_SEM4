import pandas as pd
import numpy as np
#Taking dataset from excel sheet 
file = pd.read_excel(r"C:\Users\aasri\Downloads\Lab Session1 Data.xlsx",sheet_name="Purchase data")
S = file[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].to_numpy()
C = file['Payment (Rs)'].to_numpy().reshape(-1, 1)
print(f"A = {S}")
print(f"C = {C}")

# Here we are finding dimensionality of the vector space
dimensionality = S.shape[1]
print("the dimensionality of the vector space is  = " + str(dimensionality))

#Finding total no.of vectors 
vector_space = S.shape[0]
print("the number of vectors in the vector space is = " + str(vector_space))


rank_S = np.linalg.matrix_rank(S)
print("the rank of the matrix  A is = " + str(rank_S))


pseudo_inverse = np.linalg.pinv(S)
print("the pseudo inverse of matrix A is = " + str(pseudo_inverse))

cost = np.dot(pseudo_inverse,C)
print("the cost of each product that is available for sale is  = " + str(cost.flatten()))



