# What's Next?

### Working with 1,000,000 Playlists
One idea we could use to better handle computing on the full dataset is principal component analysis.  By narrowing down our set of features we could train models faster and experiment more with model tuning.  We did create a visualization to see how we can differentiate between `match = 0` and `match = 1` tracks using the first 2 principal components.  We can see that even just two components with 41,000 playlists loaded start to discriminate between our two target categories.  It would be fascinating to think more about how this works and what the right number of components would be for an accurate model that is also easier to train on a large percentage of the data.

**Principal Component Analysis code:**
```python
#Can't do sklearn PCA on sparse matrix object, use TruncatedSVD instead
from sklearn.decomposition import TruncatedSVD
pca_transformer = TruncatedSVD(2).fit(X_train) 
X_train_2d = pca_transformer.transform(X_train)
X_test_2d = pca_transformer.transform(X_test)
```

**Plot**
*Credit to HW4 for lab for code template*
```python
colors = ['b', 'r']
label_text = ["Not Match", "Match"] 
plt.figure(figsize = (10,6))
for cur_quality in [0, 1]:
    cur_df = X_train_2d[y_train == cur_quality] 
    plt.scatter(
        cur_df[:, 0],
        cur_df[:, 1], 
        c=colors[cur_quality], 
        label=label_text[cur_quality])
plt.xlabel("PCA Dimension 1", fontsize = 13)
plt.ylabel("PCA Dimention 2", fontsize = 13)
plt.title("Scatter plots of top 2 PCA Components for Feature Data", fontsize = 15) 
plt.tick_params(labelsize = 13)
plt.grid(alpha = 0)
plt.legend();
```
![fig1](images/pca.png)

### Separating a Validation Set and Tuning Hyperparameters

### Taking advantage of the Million Song Dataset

### Leveraging Spotify's API
Create our own playlists with the API...
