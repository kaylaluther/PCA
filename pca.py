"""
Principal Component Analysis or PCA is a widely used technique for dimensionality reduction
of the large data set. Reducing the number of components or features costs some accuracy and 
on the other hand, it makes the large data set simpler, easy to explore and visualize
"""

import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt 

#Reads in the file
df = pd.read_csv('iris.csv', header=None, sep=',') 
#defines the column names 
df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class'] 
#will print out if any of the values are null
print(df.isnull().values.any()) 

#Print Last five rows of the dataset. 
print(df.tail())

#X array will store the independent features of the flowers 
#such as the petal length, petal width, sepal length, and sepal width
X = df.iloc[:,0:4].values 
#Y array will store the target values which is the class column  which is the flower name
Y = df.iloc[:,4].values

#Now the data set is stored in a 150×4 matrix where the columns are the different features of the iris', 
#and each row represents a separate flower sample. 
#Each sample row x is in a 4-dimensional vector
print(X)
print("\n")
print(Y)
#Standardization is needed before performing PCA because PCA is very sensitive to variances. 
#Meaning, if there are large differences between the scales (ranges) of the features, 
#then those with larger scales will dominate over those with the small scales.

#Here the X_std values will be standardized to the range of -1 to +1
#this will standardized it by using a formula of 
#u is the mean of the training samples or zero, and s is the standard deviation of the training samples or one.
#z = (x - u) / s
X_std = StandardScaler().fit_transform(X)

print(X_std)

#The Eigenvalues explain the variance of the data along the new feature axes.
#The corresponding eigenvalue will tell us how much variance is included in that new transformed feature

#Eigen decomposition on the covariance matrix Σ, which is a d×d matrix where each element represents the covariance 
#between the two features. 
#d is the number of original dimensions of the data set.
#in our dataset we have 4 different features, so the covariance matrix should be a 4x4

#This will calculate the mean of the standardized vector. 
mean_vec = np.mean(X_std, axis=0) 
#This will create the instantce of a covariance matrix. The covarariance matrix is a nxn symmetric matrix.
#n will be the number of dimensions.
#The matrix will have all the possible pairs of the initial variables.

#The upper left and lower right represent the variance of the x and y variables, respectively, 
#while the identical numbers on the lower left and upper right represent the covariance between x and y.
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1) 
print('Covariance matrix \n%s' %cov_mat) 

#Here will calculate the Eigenvectors and Eigenvalues of the standardized feature values
#If there is a positive value it means that the two variables increase or decrease together, which means that they are correlated.
#If there is a negative value means as one variables increases, the other decreases, which means they are inversely correlated.
cov_mat = np.cov(X_std.T) 
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat) 
print('Eigenvectors \n%s' %eigen_vecs) 
print('\nEigenvalues \n%s' %eigen_vals)

#We know that the sum of the square of each value in an Eigenvector is 1. 
#So this will determine if the values were calculated correctly 

#This will append all of the eigen_vecs (that we stored in the vector above) to the sq_eig vector 
square_eigens=[] 
for i in eigen_vecs: 
    square_eigens.append(i**2) 
print(square_eigens) 
 
#This is going to sum all the values together in the vector and they should all equal 1
print("sum of squares of each values in an eigen vector is \n") 
print(sum(square_eigens))

for ev in eigen_vecs: 
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
    
#I will Make a list of (eigenvalue, eigenvector) and print them out
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i])
for i in range(len(eigen_vals))]
#This will pring out the eigenvalues and the array of eigenvectors
print(type(eigen_pairs)) 
#Sort the (eigenvalue, eigenvector) tuples from high to low eigen_pairs.sort() 
eigen_pairs.reverse() 
print("\n",eigen_pairs) 
#You can Visually confirm that the list is correctly sorted by decreasing eigenvalues 
print('\n\n\nEigenvalues in descending order:') 
for i in eigen_pairs: 
    print(i[0])

#in this section it will determine how manu principal components that we are going to choose for our
#new subspace.
#This explained variance will be calculated from the eigenvalues and tell us how much variance can
#can be attributed to each of the principal components.  
tot = sum(eigen_vals) 
print("\n",tot) 
var_exp = [(i / tot)*100 
for i in sorted(eigen_vals, reverse=True)] 
print("\n\n1. Variance Explained\n",var_exp) 
cum_var_exp = np.cumsum(var_exp) 
print("\n\n2. Cumulative Variance Explained\n",cum_var_exp) 
print("\n\n3. Percentage of variance the first two principal components each contain\n ",var_exp[0:2]) 
print("\n\n4. Percentage of variance the first two principal components together contain\n",sum(var_exp[0:2]))

#This will create the projection matrix from the selected eigenvectors and transform the dataset 
#from a 4-dimensional space 2-dimensional space. 
print(eigen_pairs[0][1]) 
print(eigen_pairs[1][1]) 
matrix_w = np.hstack((eigen_pairs[0][1].reshape(4,1), eigen_pairs[1][1].reshape(4,1))) 
#hstack: Stacks arrays in sequence horizontally (column wise). 
print('Matrix W:\n', matrix_w)

#This will print the new 2d sapce matrix
y = X_std.dot(matrix_w) 
principalDf = pd.DataFrame(data = y , columns = ['principal component 1', 'principal component 2']) 
print(principalDf.head())

#THis will combined the species with the principal components
finalDf = pd.concat([principalDf,pd.DataFrame(Y,columns = ['species'])], axis = 1) 
print(finalDf.head())

#This will plot the 2 components on a graph so that it will be easier to visualize
fig = plt.figure(figsize = (8,5)) 
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15) 
ax.set_ylabel('Principal Component 2', fontsize = 15) 
ax.set_title('2 Component PCA', fontsize = 20) 
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'] 
colors = ['r', 'g', 'b'] 
for target, color in zip(targets,colors): 
    indicesToKeep = finalDf['species'] == target  
ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'] , 
finalDf.loc[indicesToKeep, 'principal component 2'] , c = color , s = 50)
ax.legend(targets)
ax.grid()

"""
There are python libriaries that can calculate the PCA without having to do all the calculations

pca = PCA(n_components=2) 
# Here we can also give the percentage as a paramter to the PCA function as pca = PCA(.95). .95 means that we want to include 95% of the variance. Hence PCA will return the no of components which describe 95% of the variance. However we know from above computation that 2 components are enough so we have passed the 2 components.
principalComponents = pca.fit_transform(X_std) 
principalDf = pd.DataFrame(data = principalComponents , columns = ['principal component 1', 'principal component 2'])
principalDf.head(5) # prints the top 5 rows

finalDf = pd.concat([principalDf, finalDf[['species']]], axis = 1) 
finalDf.head(5)
"""