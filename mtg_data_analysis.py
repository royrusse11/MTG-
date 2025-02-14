import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("cards.csv")

# Display the first few rows
print(df.head())

# Show column names to understand the data structure
print("Columns in dataset:", df.columns)

# ---- STEP 7: Clean & Format Data ----

# Select only the relevant columns (ensure correct names)
df = df[['name', 'cmc', 'type_line', 'color_identity', 'power', 'toughness', 'oracle_text', 'set']]

# Fill missing values
df.fillna("N/A", inplace=True)

# Convert Color Identity list into a single string
df['color_identity'] = df['color_identity'].apply(lambda x: ','.join(x) if isinstance(x, list) else str(x))

# Convert Power & Toughness to numeric
df['power'] = pd.to_numeric(df['power'], errors='coerce').fillna(0)
df['toughness'] = pd.to_numeric(df['toughness'], errors='coerce').fillna(0)

# Show cleaned data
print("Cleaned Data Preview:")
print(df.head())

# Save cleaned data
df.to_csv("cleaned_cards.csv", index=False)
print("Cleaned dataset saved as 'cleaned_cards.csv'")

# Histogram of CMC (converted mana cost)
plt.figure(figsize=(8,5))
plt.hist(df['cmc'], bins=10, color='blue', edgecolor='black')
plt.xlabel('Converted Mana Cost (CMC)')
plt.ylabel('Number of Cards')
plt.title('Distribution of Card CMC')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Count the number of cards for each color identity
color_counts = df['color_identity'].value_counts()

# Bar chart
plt.figure(figsize=(8,5))
color_counts.plot(kind='bar', color=['white', 'blue', 'black', 'red', 'green', 'purple'])
plt.xlabel('Color Identity')
plt.ylabel('Number of Cards')
plt.title('Card Distribution by Color Identity')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Count the number of cards for each type
type_counts = df['type_line'].value_counts().head(10)  # Show only the top 10 types

# Bar chart
plt.figure(figsize=(10,5))
type_counts.plot(kind='bar', color='orange', edgecolor='black')
plt.xlabel('Card Type')
plt.ylabel('Number of Cards')
plt.title('Top 10 Most Common Card Types')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Filter out non-creatures (power/toughness = 0)
creatures = df[(df['power'] > 0) & (df['toughness'] > 0)]

# Scatter plot
plt.figure(figsize=(8,5))
plt.scatter(creatures['power'], creatures['toughness'], alpha=0.5, color='green')
plt.xlabel('Power')
plt.ylabel('Toughness')
plt.title('Creature Power vs. Toughness')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Encode categorical columns
encoders = {}
for col in ['color_identity', 'type_line', 'oracle_text', 'set']:
    df[col] = df[col].astype(str)  # Ensure they are strings
    encoders[col] = LabelEncoder()
    df[col + '_encoded'] = encoders[col].fit_transform(df[col])

# Save mapping of encoded values for easy decoding later
color_identity_mapping = dict(zip(encoders['color_identity'].classes_, encoders['color_identity'].transform(encoders['color_identity'].classes_)))
df['color_identity_decoded'] = df['color_identity_encoded'].map({v: k for k, v in color_identity_mapping.items()})

# Drop old text-based columns
df = df.drop(columns=['color_identity', 'type_line'])

# Save processed data
df.to_csv("processed_cards.csv", index=False)
print("Processed dataset saved as 'processed_cards.csv'")
