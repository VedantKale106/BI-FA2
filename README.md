# 🛒 Amazon Reviews Sentiment Analysis

A comprehensive Business Intelligence project analyzing customer sentiment from Amazon Fine Food Reviews using text mining and data visualization techniques to extract actionable business insights .



## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Team Members](#team-members)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [Business Intelligence Insights](#business-intelligence-insights)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)

## 🎯 Project Overview

This project focuses on sentiment analysis of Amazon customer reviews to help e-commerce platforms improve search relevance, enhance product recommendations, and reduce customer dissatisfaction. Through comprehensive text mining and data visualization, we extract meaningful patterns from customer feedback to drive business decisions.



## 👥 Team Members

| Roll No. | Name | Contribution |
|----------|------|-------------|
| 122B1B122 | Aaditesh Kadu | Data Analysis & Visualization |
| 122B1B124 | Vedant Kale | Sentiment Classification & EDA |
| 122B1B139 | Prajwal Khobragade | Business Intelligence Insights |

**Under Guidance of:** Pooja Bidwai Madam 

## 🔍 Problem Statement

E-commerce platforms face critical challenges including:
- Irrelevant search results leading to poor user experience
- Ineffective product recommendations 
- Negative customer experiences causing cart abandonment
- Reduced sales due to customer dissatisfaction 

## 📊 Dataset

**Source:** Amazon Fine Food Reviews (Kaggle) 
**Size:** 568,427 reviews
**Key Fields:**
- `Text`: Customer review content
- `Score`: Rating (1-5 stars)
- `Summary`: Review summary
- `ProductId`: Unique product identifier
- `UserId`: Customer identifier

## 🔬 Methodology

### 1. Data Preprocessing
```
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)  # remove non-alpha
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

### 2. Sentiment Classification
```
def score_to_sentiment(score):
    if score <= 2:
        return "Negative"
    elif score == 3:
        return "Neutral"
    else:
        return "Positive"
```

### 3. Analysis Techniques
- **Text Mining:** Tokenization, stopword removal, TF-IDF vectorization 
- **Sentiment Analysis:** Score-based classification 
- **Data Visualization:** Distribution plots, word clouds, correlation analysis 
- **Statistical Analysis:** Review length analysis, product performance metrics 

## 📈 Key Findings

### Sentiment Distribution
| Sentiment | Count | Percentage |
|-----------|-------|------------|
| Positive | 443,777 | 78.0% |
| Negative | 82,012 | 14.4% |
| Neutral | 42,638 | 7.6% |


### Review Patterns
- **Average Review Length:** Negative reviews tend to be longer than positive ones
- **Top Keywords in Negative Reviews:** "taste", "flavor", "product", "packaging", "box" 
- **Most Reviewed Products:** Analysis of top 10 products by review volume 

## 💡 Business Intelligence Insights

### 🔍 Improving Search & Recommendations
**Problem:** Customers face irrelevant search results and poor product recommendations 

**Solutions:**
1. **Priority Ranking:** Rank products with higher positive review volumes first in search results 
2. **Personalized Recommendations:** Combine user behavior with review sentiment for better product suggestions

### 🛠️ Resolving Customer Issues
**Problem:** Product and delivery issues leading to customer frustration 

**Solutions:**
1. **Quality Control:** Address recurring issues with taste, flavor, and product quality 
2. **Packaging Improvements:** Fix delivery and packaging problems identified in negative feedback 

### 📊 Monitoring & Analytics
- Implement sentiment trend monitoring to measure improvements 
- Track customer satisfaction metrics over time 

## 🛠️ Technologies Used

### Core Libraries
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
```

### Tools & Platforms
- **Programming Language:** Python 3.x
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, WordCloud
- **Machine Learning:** Scikit-learn
- **Development Environment:** Jupyter Notebook

## 🚀 Installation

1. **Clone the repository:**

2. **Install required packages:**
```
pip install pandas numpy matplotlib seaborn wordcloud scikit-learn
```

3. **Download the dataset:**
```
# Download Amazon Fine Food Reviews from Kaggle# Place 'Reviews.csv' in the project directory 

## 💻 Usage

### Run the Analysis
jupyter notebook EDA.ipynb
```

## 📊 Results

### Visualizations Generated
1. **Distribution Charts:** Rating and sentiment distribution 
2. **Word Clouds:** Negative review keywords visualization
3. **Product Performance:** Top products analysis 

### Business Impact
- **Search Improvement:** 78% positive reviews provide strong quality signals 
- **Issue Identification:** Clear patterns in negative feedback for targeted improvements 
- **Customer Experience:** Data-driven approach to enhance user satisfaction 

## 🔮 Future Work

- **Advanced Models:** Implement BERT and transformer-based sentiment analysis
- **Multilingual Support:** Extend analysis to non-English reviews 
- **Real-time Dashboards:** Integrate with live business intelligence platforms
- **Predictive Analytics:** Forecast customer satisfaction trends
- **Deep Learning:** Implement neural networks for nuanced sentiment detection

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 📞 Contact

- **Aaditesh Kadu**
- **Vedant Kale** 
- **Prajwal Khobragade** 

## 🙏 Acknowledgments

- **Dataset:** Amazon Fine Food Reviews (Kaggle)
- **Guidance:** Prof. Pooja Bidwai
---

⭐ **Star this repository if you found it helpful!** ⭐
