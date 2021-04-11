from surprise import SVD
from surprise import KNNBaseline
from surprise import Reader
from surprise import Dataset
from surprise.model_selection import cross_validate

#remove headers for csv file first
reader = Reader(line_format='user item rating timestamp', sep=',')
#Change to your own folder.
data = Dataset.load_from_file('D:/HWData/moviedataset/ratings_small.csv', reader=reader)


#for algorithm in [SVD(biased=False)]:#, KNNBaseline(sim_options=sim_options_user), KNNBaseline(sim_options=sim_options_item)]:
algo = SVD(biased=False)
print('PMF')
results = cross_validate(algo, data, measures=['MAE','RMSE'], cv=5, verbose=True)
 
for flag in [True, False]:
    for method in ['cosine', 'msd', 'pearson']:
        sim_options = {'name': method,
                  'user_based': flag
                   }
        algorithm = KNNBaseline(sim_options=sim_options)
        print('User:',flag,'Similarity Method:',method)
        results = cross_validate(algorithm, data, measures=['MAE','RMSE'], cv=5, verbose=True)
#Question F, G
for flag in [True, False]:
    for number in [1,3,5,10,30,50,100]:
        sim_options = {'name': 'msd',
                  'user_based': flag
                   }
        print('User:',flag,'K:',number)
        algorithm = KNNBaseline(sim_options=sim_options)
        results = cross_validate(algorithm, data, measures=['MAE','RMSE'], cv=5, verbose=True) 
        




