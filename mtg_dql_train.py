import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random
import gym

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

# Define Environment for Deck Building
class MTGDeckBuildingEnv(gym.Env):
    def __init__(self, card_data):
        self.card_data = card_data
        self.deck = []
        self.max_deck_size = 100  # Commander deck size
        self.commander = None
        self.commander_color_identity = None
        self.lands_needed = 20  # Fixed number of lands
        
        self.action_space = gym.spaces.Discrete(len(card_data))
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(card_data.shape[1],), dtype=np.float32)

    def reset(self):
        self.deck = []
        self.commander = None
        self.commander_color_identity = None
        return np.zeros(self.card_data.shape[1])

    def step(self, action):
        card = self.card_data.iloc[action]
        
        if len(self.deck) == 0:  # Pick the first card as the Commander
            type_line = str(card['type_line_encoded'])
            if "Legendary Creature" in type_line or "Planeswalker" in type_line:
                self.commander = card
                self.commander_color_identity = card['color_identity_decoded']
                self.deck.append(card)
                reward = 5  # Higher reward for choosing a valid commander
            else:
                reward = -5  # Penalize invalid commander choice
        else:
            # Ensure all future picks match the Commander's color identity
            if self.commander_color_identity and card['color_identity_decoded'] in self.commander_color_identity:
                if len(self.deck) < (self.max_deck_size - self.lands_needed):
                    self.deck.append(card)
                    reward = 1
                else:
                    reward = -1  # Penalize unnecessary non-land picks
            else:
                reward = -5  # Penalize off-color picks
        
        # Stop picking non-land cards after 80 cards
        if len(self.deck) >= (self.max_deck_size - self.lands_needed):
            # Add lands that match the Commander's color identity
            lands = self.card_data[(self.card_data['type_line_encoded'].astype(str).str.contains("Land")) & (self.card_data['color_identity_decoded'].isin(list(self.commander_color_identity)))]
            land_indices = random.sample(lands.index.tolist(), min(self.lands_needed, len(lands)))
            for idx in land_indices:
                self.deck.append(self.card_data.iloc[idx])
            done = True  # Deck is complete
        else:
            done = False
        
        next_state_numeric = card.drop(['name', 'color_identity_decoded']).to_numpy(dtype=np.float32)
        return np.nan_to_num(next_state_numeric, nan=0.0), reward, done, {}

# Initialize AI Model
input_size = df_numeric.shape[1]
output_size = len(df)
dqn = DQN(input_size, output_size)
optimizer = optim.Adam(dqn.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

epsilon = 1.0  # Exploration factor
epsilon_decay = 0.995
epsilon_min = 0.1
gamma = 0.9  # Discount factor

env = MTGDeckBuildingEnv(df)
num_episodes = 500  # Training cycles

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad():
                action = torch.argmax(dqn(state_tensor)).item()

        next_state, reward, done, _ = env.step(action)
        
        state_tensor = torch.tensor(state, dtype=torch.float32)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
        q_values = dqn(state_tensor)
        next_q_values = dqn(next_state_tensor)
        target_q_value = reward + gamma * torch.max(next_q_values)
        loss = loss_fn(q_values[action], target_q_value)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_reward += reward
        state = next_state

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    if episode % 10 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")

# Save trained model
torch.save(dqn.state_dict(), "mtg_deckbuilder_dqn.pth")
print("Model trained and saved as 'mtg_deckbuilder_dqn.pth'")
