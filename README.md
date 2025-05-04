# 🎬 Movie Recommendation System (Content-Based)

This is a simple movie recommendation system that suggests similar movies based on **theme, story, cast, genre, and director**. It uses natural language processing techniques and cosine similarity.

## 📁 Dataset
- **Name**: IMDB Movie Dataset (1951–2023)
- **Format**: CSV
- **Columns used**: `overview`, `genre`, `cast`, `director`, `movie_name`

🚀 Features
- Recommends movies based on theme, story, cast, and genre
- Uses TF-IDF and cosine similarity
- Built using Python and pandas

## 🔍 How It Works
- Missing values are handled.
- Text features (`overview`, `genre`, `cast`, `director`) are combined.
- TF-IDF vectorization is applied to convert text into numerical features.
- Cosine similarity is used to recommend top 10 similar movies.

## 📌 Technologies Used
- Python
- pandas
- scikit-learn (TfidfVectorizer, cosine_similarity)

## 🚀 Running the Code
This project is built and tested on **Google Colab**.

1. Open the notebook in [Google Colab](https://colab.research.google.com/)
2. Upload the CSV dataset file (`IMDB-Movie-Dataset(2023-1951).csv`)
3. Run all cells
4. Enter a movie name to get recommendations

## 🧠 Example
```python
get_recommendations('Zindagi Na Milegi Dobara')
