



https://airlinepassengersatisfaction-jjdy39kd5azftgtecurhgf.streamlit.app/

I downloaded the dataset from Kaggle and read the CSV file using the Pandas library. Then I prepared the data with EDA steps like checking for null values and renaming columns.
 
Next, I followed the ML pipeline steps:
 
1. Split the data into inputs and output
2. Split into train and test sets (80:20)
3. Feature engineering: One-Hot Encoding for categorical columns, normalization (MinMaxScaler) for numerical columns, and Label Encoding for the output (Split-Apply-Combine)
4. Trained the model with the KNN algorithm
5. Applied the same feature engineering to the test data
6. Made predictions with the model
7. Evaluated the model: ~93% accuracy
After this, I saved the model using joblib, built the frontend with Streamlit, and deployed it on Streamlit Community Cloud (Streamlit hub).
