import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Load processed card data
df = pd.read_csv("processed_cards.csv")

# Ensure we use the same number of columns as the trained model
numeric_columns = df.select_dtypes(include=[np.number]).columns
df_numeric = df[numeric_columns]

# Define the same DQN model
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Load trained model
input_size = df_numeric.shape[1]  # Match training input size
output_size = len(df)

dqn = DQN(input_size, output_size)
dqn.load_state_dict(torch.load("mtg_deckbuilder_dqn.pth"), strict=False)
dqn.eval()

# Generate a 100-card Commander deck
deck_indices = []
state = np.zeros(input_size)

for _ in range(100):  # Commander deck size
    state_tensor = torch.tensor(state, dtype=torch.float32)
    with torch.no_grad():
        q_values = dqn(state_tensor)
        action = torch.argmax(q_values).item()

    # Ensure diversity by avoiding duplicates
    while action in deck_indices:
        action = np.random.randint(0, len(df))

    deck_indices.append(action)

# Retrieve full details for selected cards
deck_details = df.iloc[deck_indices][['name', 'cmc', 'oracle_text', 'set', 'type_line_encoded', 'color_identity_decoded']]


# Save to CSV
deck_details.to_csv("ai_generated_commander_deck.csv", index=False)

print("AI-Generated Commander deck saved as 'ai_generated_commander_deck.csv'")
