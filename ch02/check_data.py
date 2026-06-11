import pandas as pd
import os

base = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base, 'input_data', 'SP98H.txt')

# Read without assuming headers
df = pd.read_csv(data_path, nrows=5, header=None)
print("Without header:")
print(df)

# Read with headers
df2 = pd.read_csv(data_path, nrows=5)
print("\nWith header:")
print(df2)
print(f"\nColumns: {df2.columns.tolist()}")