import requests
from newspaper import Article
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import streamlit as st


def search_and_scrape_articles(query, num_results):         #num_results : How much we really want
    api_key = "86ccf87d76ba262f7c5b5e361d955168880500d764ec05122ff63782bbe56de6"
    search_url = "https://serpapi.com/search.json"
    overfetch = 10                        # Try fetching more than needed

    params = {
        "q": query,
        "api_key": api_key,
        "engine": "google",
        "tbm": "nws",
        "num": overfetch
    }

    response = requests.get(search_url, params=params)
    if response.status_code != 200:
        return []

    results = response.json()
    news_results = results.get("news_results", [])

    articles = []
    for news in news_results:
        link = news.get("link")
        scraped = scrape_article_details(link)

        if scraped is not None:
            articles.append({
                "title": scraped["title"],
                #"text": scraped["text"],
                "date": scraped["date"],
                "media_house": news.get("source"),
                "original_link": link
            })

        if len(articles) >= num_results:
            break  # ✅ Stop as soon as we have enough

    return articles



def scrape_article_details(url):
    try:
        article = Article(url)
        article.download()
        article.parse()

        # for idx, article in enumerate(articles, 1):
        #     st.markdown(f"{idx}. [{article['title']}]({article['original_link']})")

        
        if article.publish_date:
            date_str = article.publish_date.strftime("%b-%y")
        else:
            date_str = "Jan-20"

        media_house = article.source_url or "Unknown"

        return {
            "title": article.title,
            #"text": article.text,
            "date": date_str,
            "media_house": media_house
        }

    except Exception as e:
        #print(f"❌ Failed to scrape {url}: {e}")
        return None


def extract_date_features(month_year_str):
    try:
        date = pd.to_datetime(month_year_str, format='%b-%y')
        return pd.Series({
            'year': date.year,
            'month': date.month,
            'quarter': date.quarter
        })
    except:
        return pd.Series({'year': 0, 'month': 0, 'quarter': 0})


def preprocess_input(statement, web, date_str, flag):
    # Step 1: BERT Embeddings
    if flag ==0:
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        bert_embeddings = model.encode([statement])

        # Step 2: Date Features
        date_features = extract_date_features(date_str)
        date_features_filled = date_features.fillna(0).values.reshape(1, -1)

        # Step 3: One-Hot Encode Media Source
        web_encoder = joblib.load("web_encoder_fake.pkl")
        web_encoded = web_encoder.transform([[web]])

        # Step 4: Combine features
        final_features = np.hstack([
            bert_embeddings,
            web_encoded,
            date_features_filled
        ])

        final_features = np.nan_to_num(final_features, nan=0.0)

        # Step 5: Normalize
        scaler = joblib.load("scaler_fake.pkl")
        final_features_scaled = scaler.transform(final_features)

        return final_features_scaled

    else:
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        bert_embeddings = model.encode([statement])
        #print(bert_embeddings)

        # STEP 2.5: Apply Date Feature Extraction
        date_features = extract_date_features(date_str)

        # STEP 3: Source (Web) Encoding
        web_encoder_stack = joblib.load("web_encoder_stack.pkl")
        web_encoded = web_encoder_stack.transform([[web]])  # double brackets to create 2D shape

        # Step 1: Replace NaNs in date features
        date_features_filled = date_features.fillna(0)
        date_features_array = date_features_filled.values.reshape(1, -1)

        # Step 3: Concatenate again
        final_features = np.hstack([
            bert_embeddings,
            web_encoded,
            date_features_array
        ])

        # Step 4: Replace any remaining NaNs (if any) as safety net
        final_features = np.nan_to_num(final_features, nan=0.0)
        #print(final_features)

        scaler_stack = joblib.load("scaler_category_stack.pkl")
        final_features_scaled = scaler_stack.transform(final_features)

        return final_features_scaled

#(NewsTitle, flag)
def start_web_search(NewsTitle, flag):
    # flag = 1
    # NewsTitle = "26/11 Plotter Tahawwur Rana Sent To Anti-Terror Agency's Custody For 18 Days"
    news_title = NewsTitle
    num_articles = 7

    print("\n🔍 Starting article lookup and processing...")
    articles = search_and_scrape_articles(news_title, num_results=num_articles)

    if not articles:
        print("❌ No articles found or all failed to process.")
        return

    vectors = []
    for idx, article in enumerate(articles):

        try:
            vec = preprocess_input(article["title"], article["media_house"], article["date"], flag)
            vectors.append(vec)
            # print(f"✅ Processed article {idx+1}: {article['title']}")
        except Exception as e:
            print(f"⚠️ Preprocessing error for article {idx+1}: {e}")
            pass

    if not vectors:
        print("❌ No vectors were created successfully.")
        return

    final_vector = np.vstack(vectors)
    #print(final_vector.shape)
    return final_vector
    #print(final_vector.shape)
    #print(final_vector.shape[0])
    # OPTIONAL: Use model to predict
    # model = joblib.load("your_model_path.pkl")
    # predictions = model.predict(final_vector)
    # print("\n🧠 Predictions:", predictions)


# Entry point
if __name__ == "__main__":
    start_web_search()
