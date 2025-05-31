# Restaurant-Review-Sentiment-Analysis
*   The project uses Python and various libraries for natural language processing and machine learning.
*   Its purpose is to **process restaurant reviews** and **predict their sentiment**, likely classifying them as positive or negative.
*   The input data is read from a tab-separated value file named `a2_RestaurantReviews_FreshDump.tsv`.
*   A key part of the project is **text preprocessing** applied to the restaurant reviews. This involves:
    *   Removing characters that are not letters.
    *   Converting the text to lowercase.
    *   Splitting the reviews into individual words.
    *   Removing common English words (stopwords), but notably **keeping the word 'not'** as it can change the meaning of a sentence.
    *   Reducing words to their root form using stemming (specifically the Porter Stemmer).
*   The project uses a **pre-trained Bag-of-Words model**, loaded from a file (`c1_BoW_Sentiment_Model.pkl`), to transform the preprocessed text into a numerical representation (a feature matrix).
*   A **pre-trained sentiment classifier model**, loaded from another file (`c2_Classifier_Sentiment_Model`), is used to make predictions on the numerical features. These predictions are likely binary, representing negative (0) or positive (1) sentiment.
*   The preprocessing, feature extraction, and classification steps shown in the code are specifically applied to the **first 100 reviews** from the input dataset.
*   The predicted sentiment labels are **added as a new column** ('predicted_label') to the original dataset for these reviews.
*   The modified dataset, including the predicted sentiments for the first 100 reviews, is then **saved to a new tab-separated file** named `c3_Predicted_Sentiments_Fresh_Dump.tsv`.
*   Finally, the project **visualizes the counts** of the predicted negative (0) and positive (1) sentiments among the processed reviews using a bar chart generated with Matplotlib. The bar chart is titled 'Sentiment Prediction Counts' and shows the distribution of predicted sentiments.
