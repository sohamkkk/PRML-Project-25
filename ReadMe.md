
# 📰 News Authentication & Domain Classifier
A machine learning-powered tool designed to verify the authenticity of news articles and classify them into appropriate domains (e.g., Politics, COVID-19, Government Affairs, etc.).


## 📌 Project Objective

To develop a robust system that:

- Authenticates news: Identifies whether a news article is true or false.

- Classifies domain: Assigns the news to a relevant topic category.

- Leverages web search and semantic understanding for human-like validation.
## 🧠 Key Features

- Checks if a news article is **real or fake** using smart machine learning models  
- Identifies the **topic** of the news (like Politics, COVID-19, Government, etc.)  
- Uses **Google Search (via [SerpApi](https://serpapi.com/))** to find similar articles for better accuracy  
- Applies **BERT**, a powerful language model, to understand the meaning of news  
- Compares different machine learning models like SVM, Decision Tree, ANN, etc.  
- Uses a **combination of the best models** (SVM + Decision Tree + ANN) to improve topic detection  
- Designed to work in a way that **mimics human judgment** by checking real-time news articles online
- 
## 📌 How to run

- Zero step --> Clone the repository using either HTTPS or SSH.
  Then in the terminal -->
- First run --> pip install -r requirements.txt
- Then start the server with --> streamlit run app2.py


## 🛠️ Tech Stack

- **Programming Language**: Python  
- **Frontend Interface**: Streamlit  
- **Deployment Platform**: Google Cloud Platform (GCP)  
- **Machine Learning & NLP**: Used some pre-built libraries like scikit-learn, TensorFlow/Keras, XGBoost, Transformers for model training and language understanding  
- **Web Search**: SerpApi for fetching real-time news articles  
- **Dataset**: Indian Fake News Dataset [IFND](https://www.kaggle.com/datasets/sonalgarg174/ifnd-dataset)
## 🙏 Acknowledgements

We would like to thank the following sources and contributors whose work helped shape our project:

- **Sonal Garg** for providing the [IFND Dataset](https://www.kaggle.com/datasets/sonalgarg174/ifnd-dataset) on Kaggle  
- **Z Khanam, B N Alwasel, H Sirafi, and M Rashid** for their research paper: *[Fake News Detection Using Machine Learning Approaches](https://iopscience.iop.org/article/10.1088/1757-899X/1099/1/012040)*  
- **Soha Mohajeri** for the [BuzzFeed News Classification notebook](https://www.kaggle.com/code/sohamohajeri/buzzfeed-news-analysis-and-classification/notebook) on Kaggle  
- Special thanks to our dedicated course instructor [Anand Mishra](https://anandmishra22.github.io/) for their invaluable guidance throughout the project.

## Authors

- [@Sahilpreet Singh(B23CS1061)](https://github.com/sps1001)
- [@Soham Khairnar (B23CM1039](https://github.com/sohamkkk)
- [@Abhishek Garg (B23EE1081)](https://github.com/ABHIGGGG)
- [@Rudra Thakar (B23EE1100)](https://github.com/rudraThakar)
- [@Kartik Gehlot (B23EE1088)](https://github.com/B23BB1024)
- [@Raman Pareekh (B23EE1091)](https://github.com/Brainloft05)


