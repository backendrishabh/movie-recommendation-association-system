from flask import Flask, render_template, request
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

app = Flask(__name__)

# Load and preprocess data
def load_data():
    df = pd.read_csv('data.csv')
    all_movies = df['Movies'].str.split(';')
    movie_list = pd.get_dummies(pd.DataFrame(all_movies.tolist()).stack()).groupby(level=0).sum()
    return movie_list

# Generate association rules
def generate_rules(movie_list):
    frequent_itemsets = apriori(movie_list, min_support=0.3, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    return rules

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendation = []
    movie_list = load_data()
    rules = generate_rules(movie_list)

    if request.method == 'POST':
        selected_movie = request.form['movie']
        for _, row in rules.iterrows():
            if selected_movie in row['antecedents']:
                recommendation.extend(list(row['consequents']))
        recommendation = list(set(recommendation))  # Remove duplicates

    return render_template('index.html', movies=movie_list.columns, recommendation=recommendation)

if __name__ == '__main__':
    app.run(debug=True)
