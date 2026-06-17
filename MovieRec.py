import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

#fetch data and format it
data = fetch_movielens(min_rating=4.0)

#print training and testing data
print(repr(data['train']))
print(repr(data['test']))

#create models
model1 = LightFM(loss='warp')
model2 = LightFM(loss='bpr')
model3 = LightFM(loss='warp-kos')
model4 = LightFM(loss='logistic')
#train model
model1.fit(data['train'], epochs=30, num_threads=2)
model2.fit(data['train'], epochs=30, num_threads=2)
model3.fit(data['train'], epochs=30, num_threads=2)
model4.fit(data['train'], epochs=30, num_threads=2)

def sample_recommendation(model1, model2, model3, model4, data, user_ids):

    #number of users and movies in training data
    n_users, n_items = data['train'].shape

    #generate recommendations for each user we input
    for user_id in user_ids:

        #movies they already like
        known_indices = data['train'].tocsr()[user_id].indices
        known_positives = data['item_labels'][known_indices]

        #movies our model predicts they will like
        scores1 = model1.predict(user_id, np.arange(n_items))
        scores2 = model2.predict(user_id, np.arange(n_items))
        scores3 = model3.predict(user_id, np.arange(n_items))
        scores4 = model4.predict(user_id, np.arange(n_items))

        # exclude already-rated items from recommendations
        scores1[known_indices] = -np.inf
        scores2[known_indices] = -np.inf
        scores3[known_indices] = -np.inf
        scores4[known_indices] = -np.inf

        #rank them in order of most liked to least
        top_items = [
            data['item_labels'][np.argsort(-scores1)][:3],
            data['item_labels'][np.argsort(-scores2)][:3],
            data['item_labels'][np.argsort(-scores3)][:3],
            data['item_labels'][np.argsort(-scores4)][:3],
        ]
        model_names = ['warp', 'bpr', 'warp-kos', 'logistic']

        # overall top items by the sum of all model scores
        summed_scores = scores1 + scores2 + scores3 + scores4
        top_overall_indices = np.argsort(-summed_scores)[:3]
        top_overall_items = [
            (data['item_labels'][item_id], float(summed_scores[item_id]))
            for item_id in top_overall_indices
        ]

        #print out the results
        print("User %s" % user_id)
        print("     Known positives:")

        for x in known_positives[:3]:
            print("        %s" % x)

        print("     Recommended:")

        for model_name, items in zip(model_names, top_items):
            print("        %s: %s" % (model_name, list(items)))

        print("     Top overall items:")
        for title, score in top_overall_items:
            print("        %s (score=%.4f)" % (title, score))
            
sample_recommendation(model1, model2, model3, model4, data, [3, 25, 450])



