import pandas as pd

def main():
    # Read the CSV file
    df = pd.read_csv("dataset.csv", sep=";", encoding="utf-8")

    # Display the first few rows of the DataFrame
    print(df.head())

    # Perform some basic data analysis
    print("Summary Statistics:")
    print(df.describe())

    # Check for missing values
    print("Missing Values:")
    print(df.isnull().sum())
    
    # Clean missing values
    df.dropna(inplace=True)
    print(df.isnull().sum())
    print(df.head())
    
    # Check the numbers of fraud and non-fraud cases
    print("Fraud Cases:")
    print(df[df['fraudulent'] == 1].shape[0]) # 866 cases
    print("Non-Fraud Cases:")
    print(df[df['fraudulent'] == 0].shape[0]) # 17014 cases
    
     # Check the distribution of the 'fraudulent' column
    print("Distribution of Fraudulent Cases:")
    print(df['fraudulent'].value_counts(normalize=True))
    
    # Save the cleaned DataFrame to a new CSV file
    df.to_csv("cleaned_dataset.csv", sep=";", encoding="utf-8", index=False)
    print("Cleaned dataset saved to 'cleaned_dataset.csv'.")
    
    


if __name__ == "__main__":
    main()