'''
Strategy: K-means tries to partition x data points into the set of k clusters where each data point is assigned to its closest cluster.
The algorithm clusters the data into k clusters, even if k is not the right number of clusters to use.
This answer I used 20 clusters, but optimal cluster number can be tested with different techniques like elbow, silhoutte or gap statistic methods

'''

# Dataset 20 newsgroups text dataset
dataset = fetch_20newsgroups(subset='all', categories=categories,
                           shuffle=True, random_state=42)

# compute a word occurrence frequency (sparse) matrix
vectorizer = TfidfVectorizer(max_df=0.5,
                           min_df=2,
                           stop_words='english')
X = vectorizer.fit_transform(dataset.data)

# K means clustering
km = KMeans(n_clusters=20, init='k-means++', max_iter=100, n_init=1)
km.fit(X)

# Locating centroids
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
  print("cluster %d:" % i)
  for ind in order_centroids[i,:20]:
      print('%s' % terms[ind])
  print()

# Effectiveness of clustering
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, km.labels_, sample_size=1000))  
