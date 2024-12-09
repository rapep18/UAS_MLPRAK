from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load Data
restaurants = pd.read_csv('data/Restaurants.csv')
ratings = pd.read_csv('data/ratings.csv')

# Content-Based Filtering
tfidf = TfidfVectorizer(stop_words='english')
restaurants['Reviews'] = restaurants['Reviews'].fillna('')
tfidf_matrix = tfidf.fit_transform(restaurants['Reviews'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Mapping restaurant names to indices
indices = pd.Series(restaurants.index, index=restaurants['Name']).drop_duplicates()

def get_content_recommendations(name, top_n=5):
    if name not in indices:
        return ["Restaurant not found"]
    
    idx = indices[name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]
    restaurant_indices = [i[0] for i in sim_scores]
    return restaurants['Name'].iloc[restaurant_indices].tolist()

# Collaborative Filtering
def get_collaborative_recommendations(user_id, top_n=5):
    user_ratings = ratings[ratings['userId'] == user_id]
    if user_ratings.empty:
        return ["No ratings found for this user"]
    
    top_rated = user_ratings.sort_values(by='rating', ascending=False)['ID'].head(top_n)
    recommended = restaurants[restaurants['RestID'].isin(top_rated)]['Name'].tolist()
    return recommended

# Hybrid Recommendations
def get_hybrid_recommendations(user_id, name, top_n=10):
    content_recommendations = get_content_recommendations(name, top_n)
    collaborative_recommendations = get_collaborative_recommendations(user_id, top_n)
    hybrid = list(set(content_recommendations + collaborative_recommendations))
    return hybrid[:top_n]

# Flask App
from flask import Flask, render_template, request

app = Flask(__name__)

# Route untuk halaman Landing Page (Home)
@app.route('/')
def landing_page():
    return render_template('landing.html')

# Route untuk halaman Index (Form Input)
@app.route('/index')
def index():
    return render_template('index.html')

# Route untuk menangani rekomendasi dan menampilkan hasil
@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.form['user_id'])
    restaurant_name = request.form['restaurant_name']
    
    # Fungsi untuk mendapatkan rekomendasi berdasarkan input pengguna
    recommendations = get_hybrid_recommendations(user_id, restaurant_name)
    
    # Render halaman hasil dengan daftar rekomendasi
    return render_template('result.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
