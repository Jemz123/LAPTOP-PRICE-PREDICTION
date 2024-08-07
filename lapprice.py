import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Path to your CSV file
file_path = r"C:\Users\Administrator\Desktop\pythonprojects\data.csv"

# Function to read the CSV file with automatic encoding detection
def read_csv_with_encoding(file_path):
    encodings = ['ISO-8859-1', 'cp1252', 'utf-16']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"Successfully read the file with encoding: {encoding}")
            return df
        except UnicodeDecodeError:
            print(f"Encoding error with {encoding}")
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None
        except pd.errors.EmptyDataError:
            print("File is empty")
            return None
        except pd.errors.ParserError:
            print("Error parsing the file")
            return None
    
    print("Failed to read the file with all attempted encodings.")
    return None

# Read the data
df = read_csv_with_encoding(file_path)

if df is not None:
    # Print column names to verify
    print("Column names:", df.columns.tolist())
    
    # Check for missing values and handle them if necessary
    if df.isnull().sum().sum() > 0:
        print("Missing values found. Handling missing values...")
        df = df.dropna()  # Simple handling: dropping missing values; consider other strategies if needed

    # Update column names in the dataset to match those used in the code
    df.rename(columns={'Ram': 'RAM', 'ROM': 'Storage'}, inplace=True)
    
    # Ensure correct columns are used
    required_columns = {'RAM', 'Storage', 'brand', 'processor', 'price'}
    if not required_columns.issubset(df.columns):
        print("One or more columns are missing from the dataset.")
        missing_columns = required_columns - set(df.columns)
        print(f"Missing columns: {missing_columns}")
    else:
        # Define features and target
        X = df[['brand', 'processor', 'RAM', 'Storage']]
        y = df['price']

        # Ensure 'RAM' and 'Storage' are numerical
        X['RAM'] = X['RAM'].astype(str).str.extract('(\d+)').astype(float)
        X['Storage'] = X['Storage'].astype(str).str.extract('(\d+)').astype(float)

        # Preprocessing for categorical data
        categorical_features = ['brand', 'processor']
        numeric_features = ['RAM', 'Storage']

        # Create the preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])

        # Create the pipeline with preprocessing and regression model
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=0))
        ])

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Mean Absolute Error: ${mae:.2f}")
        print(f"R^2 Score: {r2:.2f}")

        # Example of predicting the price of a new laptop
        new_data = pd.DataFrame({
            'brand': ['Apple'],
            'processor': ['Apple M1'],
            'RAM': ['16GB'],
            'Storage': ['800GB']
        })

        # Convert new data features to appropriate types
        new_data['RAM'] = new_data['RAM'].astype(str).str.extract('(\d+)').astype(float)
        new_data['Storage'] = new_data['Storage'].astype(str).str.extract('(\d+)').astype(float)

        # Predict the price for the new laptop
        predicted_price = model.predict(new_data)
        print(f"Predicted Price for new laptop: ${predicted_price[0]:.2f}")
else:
    print("Failed to load data. Please check the file and encoding.")
