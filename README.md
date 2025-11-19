# 📦 Amazon Reviews Sentiment Analysis (SVM + TF-IDF)

This project builds a **Sentiment Analysis system** that classifies Amazon product
reviews as **Positive** or **Negative** using **TF-IDF vectorization** and a
**Support Vector Machine (SVM)** classifier.

The project includes:
- A full **ML pipeline** (preprocessing → feature extraction → model training)
- A saved **SVM model + TF-IDF vectorizer**
- A fully interactive **frontend website** to test the model live
- A clean and optimized dataset pipeline for large review datasets

---

## 🚀 **Project Features**

### 🔍 **Machine Learning**
- Preprocessed large Amazon product review data  
- Removed stopwords, punctuation, and normalized the text  
- Converted text to numerical vectors using **TF-IDF**  
- Trained a **Linear SVM (Support Vector Classifier)**  
- 85–90% accuracy depending on sample size  
- Saved the final model as:
  - `svm_sentiment_model.pkl`
  - `tfidf_vectorizer.pkl`

---

## 🌐 **Interactive Website**
A clean, responsive, React-based UI where users can:

✔ Enter any review  
✔ See the cleaned text  
✔ Understand model weight contributions  
✔ View the SVM decision score  
✔ Get Positive / Negative sentiment instantly  

### 🔧 Website Features:
- Real-time sentiment prediction  
- Visual SVM decision score  
- Feature-weight breakdown  
- Mobile-friendly UI  
- Clean minimal design

---

## 🗂️ **Folder Structure**


