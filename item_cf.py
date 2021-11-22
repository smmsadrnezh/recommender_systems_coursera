import pandas as pd
from numpy import dot
from numpy.linalg import norm


def print_most_recommended_items(similarity_matrix, similarity_method, ratings, user_id):
    def predict(item_rating):
        item_id = item_rating.name
        item_similarity = similarity_matrix[similarity_method][item_id]
        return dot(user_ratings[user_id], item_similarity) / sum(item_similarity)

    user_ratings = ratings['raw'].loc[user_id, :].to_frame()
    user_ratings['predict'] = user_ratings.apply(predict, axis=1)
    user_ratings = user_ratings['predict'].sort_values(ascending=False).head(5)
    print(user_ratings)


def print_most_similar_items(similarity_matrix, rating_method, item):
    most_similar_items = similarity_matrix[rating_method][item].sort_values(ascending=False)[1:].head(5)
    print(most_similar_items)


def get_similarity_matrix(ratings):
    def get_similarity(rating_method, item_1, item_2):
        a = ratings[rating_method][item_1][:-1]
        b = ratings[rating_method][item_2][:-1]
        return dot(a, b) / (norm(a) * norm(b))

    items = ratings['raw'].columns
    items_similarity = {
        'raw': pd.DataFrame(index=items, columns=items),
        'norm': pd.DataFrame(index=items, columns=items)
    }
    for rating_method in 'raw', 'norm':
        for item_1 in items:
            for item_2 in items:
                similarity = get_similarity(rating_method, item_1, item_2)
                items_similarity[rating_method].loc[item_1, item_2] = similarity
    return items_similarity


def extract_data():
    ratings = {}
    for rating_method in 'raw', 'norm':
        df = pd.read_csv(f'datasets/{rating_method}_ratings.csv')
        df.columns = df.columns.to_series().apply(lambda x: x.split(":")[0])
        df.set_index('User', inplace=True, drop=True)
        ratings[rating_method] = df
    # users_mean = ratings['raw']['Mean']
    ratings['raw'].drop(['Mean'], axis=1, inplace=True)
    ratings['raw'].fillna(0, inplace=True)
    return ratings


def main():
    ratings = extract_data()
    similarity_matrix = get_similarity_matrix(ratings)
    # print_most_similar_items(similarity_matrix, 'raw' , '1')
    # print_most_similar_items(similarity_matrix, 'norm' , '1')
    print_most_recommended_items(similarity_matrix, 'raw', ratings, '755')
    # print_most_recommended_items(similarity_matrix, 'norm', ratings, '5277')


if __name__ == '__main__':
    main()
