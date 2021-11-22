import math
import pandas as pd


def print_answers(df, users):
    for user in users:
        for question in f'predict_wo_norm_{user}', f'predict_w_norm_{user}':
            print(df.sort_values(by=question, ascending=False).loc[:, question].iloc[:3], end='\n\n')


def add_predicts(df, corr, users):
    answers = pd.DataFrame()
    for i in users:
        neighbours = pd.DataFrame(corr.loc[i]).sort_values(by=i, ascending=False).iloc[1:6, :]

        def predict(item_rates):
            rates_sum = 0
            weights_sum = 0
            for neighbour_id, similarity in neighbours.iterrows():
                neighbour_item_rate = item_rates[neighbour_id]
                if not math.isnan(neighbour_item_rate):
                    weight = similarity.iloc[0]
                    neighbour_average_rating = df.loc[:, neighbour_id].mean() if with_normalization else 0

                    rates_sum += (neighbour_item_rate - neighbour_average_rating) * weight
                    weights_sum += weight

            return round(rates_sum / weights_sum + user_average, 3) if weights_sum != 0 else 0

        with_normalization = False
        user_average = 0
        answers[f'predict_wo_norm_{i}'] = df.apply(predict, axis=1)

        with_normalization = True
        user_average = df.loc[:, str(i)].mean()
        answers[f'predict_w_norm_{i}'] = df.apply(predict, axis=1)
    return answers


def extract_data():
    df = pd.read_csv('datasets/movies.csv')
    df['movie'] = df.apply(lambda x: x[0].split(":")[0], axis=1)
    df.drop(['name'], axis=1, inplace=True)
    df.set_index('movie', inplace=True, drop=True)
    corr = pd.read_csv('datasets/correlations.csv', index_col=0, header=0)
    return df, corr


def main():
    df, corr = extract_data()
    users = (3867, 89)
    answers = add_predicts(df, corr, users)
    print_answers(answers, users)


if __name__ == '__main__':
    main()
