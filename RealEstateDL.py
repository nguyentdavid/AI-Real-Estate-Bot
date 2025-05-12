
# importing libraries

import gym  # importing Gym for building RL environment
import numpy as np  # importing NumPy for array operations
import pandas as pd  # importing pandas for data loading and processing
import random  # using for exploration strategy
import matplotlib.pyplot as plt  # using for plotting graphs
from collections import deque  # using deque for replay buffer
import torch  # importing PyTorch for model and tensors
import torch.nn as nn  # importing neural network components
import torch.optim as optim  # importing optimizers
from sklearn.preprocessing import MinMaxScaler  # scaling features between 0 and 1
from torch.utils.data import DataLoader, Dataset  # using DataLoader and Dataset for LSTM batching


# loading and preprocessing data

real_estate_data = pd.read_csv('real_estate_data.csv', usecols=['DATE', 'HPI'])  # loading real estate HPI data
economic_indicators = pd.read_csv('economic_indicators.csv', usecols=['DATE', 'DIV', 'GDP'])  # loading economic data
real_estate_data.rename(columns={'DATE': 'Date'}, inplace=True)  # renaming date column
economic_indicators.rename(columns={'DATE': 'Date'}, inplace=True)  # renaming date column
data = real_estate_data.merge(economic_indicators, on='Date')  # merging both datasets on Date
data['Date'] = pd.to_datetime(data['Date'])  # converting to datetime format
data = data.ffill()  # forward-filling missing values

base_price = 166000  # setting base price in year 2000
base_year = '2000-01-01'  # setting base date
base_year_hpi = data.loc[data['Date'] == base_year, 'HPI'].values[0]  # getting HPI of base year
data['PropertyPrice'] = (base_price * (data['HPI'] / base_year_hpi)).round().astype(int)  # calculating prices

scaler = MinMaxScaler()  # initializing scaler
features = [col for col in data.columns if col not in ['Date', 'PropertyPrice']]  # selecting features to scale
scaled_features = scaler.fit_transform(data[features])  # applying normalization
data_scaled = pd.DataFrame(scaled_features, columns=features)  # building scaled dataframe
data_scaled['PropertyPrice'] = data['PropertyPrice']  # adding original price column back


# defining custom real estate environment

class RealEstateEnv(gym.Env):  # creating Gym environment class
    def __init__(self, data):  # initializing environment
        super().__init__()  # calling parent constructor
        self.data = data  # storing data
        self.currentStep = 0  # tracking step index
        self.balance = 1000000  # initializing cash
        self.numPropertyOwned = 0  # tracking property ownership
        self.action_space = gym.spaces.Discrete(3)  # defining 3 actions: hold, buy, sell
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.data.shape[1] - 1 + 2,), dtype=np.float32)  # defining state space

    def _get_obs(self):  # building observation vector
        obs = self.data.drop(columns=['PropertyPrice']).iloc[self.currentStep].values  # getting market data
        norm_balance = self.balance / 1_000_000  # normalizing balance
        norm_owned = self.numPropertyOwned / 5  # normalizing property count
        return np.concatenate([obs, [norm_balance, norm_owned]])  # returning full observation

    def reset(self):  # resetting environment
        self.currentStep = 0  # resetting step
        self.balance = 1000000  # resetting balance
        self.numPropertyOwned = 0  # resetting holdings
        return self._get_obs()  # returning initial state

    def step(self, action):  # performing action and returning outcome
        currentPrice = self.data.iloc[self.currentStep]['PropertyPrice']  # getting current price
        reward = 0  # initializing reward

        if action == 1 and self.balance >= currentPrice:  # handling buying
            self.numPropertyOwned += 1  # increasing properties
            self.balance -= currentPrice  # deducting from cash
            reward += 10  # rewarding buy
        elif action == 2:  # handling selling
            if self.numPropertyOwned > 0:  # checking ownership
                self.numPropertyOwned -= 1  # reducing owned
                self.balance += currentPrice  # adding cash
                reward += 5  # rewarding sell
            else:
                reward -= 1000  # penalizing invalid sell
        elif action == 0:  # handling holding
            reward += 1  # rewarding passive action

        self.currentStep += 1  # moving to next time step
        done = self.currentStep >= len(self.data) - 1  # checking if episode is over
        nextState = self._get_obs()  # getting next state

        if not done:  # if not done, calculate portfolio growth
            portfolioValue = self.balance + self.numPropertyOwned * self.data.iloc[self.currentStep]['PropertyPrice']
            reward += (portfolioValue - self.balance) * 0.001  # giving reward for growth

        return nextState, reward, done, {}  # returning results


# defining LSTM price prediction model

class PricePredictorLSTM(nn.Module):  # building LSTM model
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):  # initializing model
        super().__init__()  # calling base constructor
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)  # building LSTM stack
        self.output = nn.Sequential(nn.Linear(hidden_dim, 1))  # adding linear output

    def forward(self, x):  # forward pass
        out, _ = self.lstm(x)  # passing through LSTM
        return self.output(out[:, -1, :]).squeeze()  # using last timestep's output

# creating dataset class for LSTM training
class RealEstateDataset(Dataset):
    def __init__(self, df, window_size=10):  # initializing dataset
        self.X, self.y = [], []  # initializing storage
        for i in range(len(df) - window_size):  # building sequences
            window = df.iloc[i:i+window_size].drop(columns=['PropertyPrice']).values  # getting features
            target = (df.iloc[i + window_size]['PropertyPrice'] - df['PropertyPrice'].mean()) / df['PropertyPrice'].std()  # normalizing target
            self.X.append(window)  # storing window
            self.y.append(target)  # storing target

    def __len__(self): return len(self.X)  # returning dataset size
    def __getitem__(self, idx):  # returning indexed item
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)


# training LSTM model

window_size = 10  # setting window size
lstm_data = RealEstateDataset(data_scaled, window_size)  # initializing dataset
dataloader = DataLoader(lstm_data, batch_size=32, shuffle=True)  # batching data
input_dim = data_scaled.shape[1] - 1  # determining input size
lstm_model = PricePredictorLSTM(input_dim)  # creating LSTM model
optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)  # initializing optimizer
criterion = nn.MSELoss()  # using MSE loss

for epoch in range(50):  # running for 50 epochs
    total_loss = 0  # resetting loss
    lstm_model.train()  # setting training mode
    for X_batch, y_batch in dataloader:  # looping over batches
        optimizer.zero_grad()  # resetting gradients
        output = lstm_model(X_batch)  # predicting
        price_mean = data['PropertyPrice'].mean()  # calculating mean
        price_std = data['PropertyPrice'].std()  # calculating std
        unnormalized_output = output * price_std + price_mean  # denormalizing prediction
        unnormalized_y = y_batch * price_std + price_mean  # denormalizing target
        loss = criterion(unnormalized_output, unnormalized_y)  # computing loss
        loss.backward()  # performing backprop
        optimizer.step()  # updating weights
        total_loss += loss.item()  # adding to total
    rmse = (total_loss / len(dataloader))**0.5  # computing RMSE
    print(f"Epoch {epoch+1}, Average Price Prediction Error: ${rmse:,.0f}")  # showing error


# generating forecast for action biasing

lstm_model.eval()  # switching to eval mode
window = data_scaled.drop(columns=['PropertyPrice']).iloc[-window_size:].values.reshape(1, window_size, -1)  # preparing input
forecast = lstm_model(torch.tensor(window, dtype=torch.float32)).item()  # predicting price
forecast = forecast * data['PropertyPrice'].std() + data['PropertyPrice'].mean()  # denormalizing output
boost_buy = forecast > data_scaled.iloc[-1]['PropertyPrice'] * 1.02  # creating buy flag


# defining Q-network for DQN agent

class QNetwork(nn.Module):  # building neural network for Q-learning
    def __init__(self, state_dim, action_dim):  # taking state and action size
        super(QNetwork, self).__init__()  # calling nn.Module init
        self.fc = nn.Sequential(  # creating fully connected layers
            nn.Linear(state_dim, 128),  # first dense layer
            nn.ReLU(),  # applying ReLU
            nn.Linear(128, 128),  # second dense layer
            nn.ReLU(),  # applying ReLU again
            nn.Linear(128, action_dim)  # outputting Q-values for each action
        )

    def forward(self, x):  # defining forward pass
        return self.fc(x)  # returning output


# defining replay buffer for storing experience

class ReplayBuffer:  # creating buffer class
    def __init__(self, capacity=10000):  # setting max capacity
        self.buffer = deque(maxlen=capacity)  # initializing deque

    def push(self, *args):  # adding transition to buffer
        self.buffer.append(args)  # appending to deque

    def sample(self, batch_size):  # sampling a batch of transitions
        batch = random.sample(self.buffer, batch_size)  # selecting random samples
        return tuple(torch.tensor(x, dtype=torch.float32 if i != 1 else torch.int64)  # formatting tensors
                     for i, x in enumerate(zip(*batch)))

    def __len__(self): return len(self.buffer)  # returning current buffer size


# initializing DQN components and hyperparameters

env = RealEstateEnv(data_scaled)  # creating environment
state_dim = env.observation_space.shape[0]  # getting state dimension
action_dim = env.action_space.n  # getting number of actions
q_net = QNetwork(state_dim, action_dim)  # creating Q-network
target_net = QNetwork(state_dim, action_dim)  # creating target network
target_net.load_state_dict(q_net.state_dict())  # copying weights
buffer = ReplayBuffer()  # creating replay buffer
optimizer = optim.Adam(q_net.parameters(), lr=0.001)  # setting optimizer

# setting hyperparameters
gamma = 0.99  # discounting future rewards
epsilon = 1.0  # starting exploration rate
decay = 0.995  # decaying epsilon
min_epsilon = 0.05  # minimum exploration
batch_size = 64  # batch size for training
episode_rewards, episode_rois, episode_values, plotPoints = [], [], [], []  # tracking metrics


# training DQN agent with environment
print("Starting value of all episodes is $1,000,000")
for episode in range(1000):  # looping over episodes
    state = env.reset()  # resetting environment
    total_reward, buy, hold, sell = 0, 0, 0, 0  # initializing counters

    for t in range(500):  # limiting steps per episode
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # preparing state input

        if random.random() < epsilon:  # selecting random action
            action = env.action_space.sample()
        else:  # selecting best action from Q-network
            with torch.no_grad():
                q_values = q_net(state_tensor)  # getting Q-values
                action = q_values.argmax().item()  # selecting best action

        next_state, reward, done, _ = env.step(action)  # stepping environment
        buffer.push(state, action, reward, next_state, float(done))  # storing experience
        state = next_state  # updating state
        total_reward += reward  # accumulating reward

        # tracking action counts
        if action == 0: hold += 1
        elif action == 1: buy += 1
        elif action == 2: sell += 1

        if len(buffer) >= batch_size:  # training if buffer has enough data
            s, a, r, ns, d = buffer.sample(batch_size)  # sampling batch
            q_val = q_net(s).gather(1, a.unsqueeze(1)).squeeze()  # calculating Q(s,a)
            with torch.no_grad():
                q_next = target_net(ns).max(1)[0]  # getting max Q(s', a')
            target = r + gamma * (1 - d) * q_next  # computing target
            loss = nn.MSELoss()(q_val, target)  # computing loss
            optimizer.zero_grad()  # resetting gradients
            loss.backward()  # backpropagating
            optimizer.step()  # updating weights

        if done: break  # stopping episode if done

    epsilon = max(min_epsilon, epsilon * decay)  # decaying epsilon
    if episode % 10 == 0:  # updating target network periodically
        target_net.load_state_dict(q_net.state_dict())

    final_price = env.data.iloc[env.currentStep]['PropertyPrice']  # getting final price
    final_value = env.balance + env.numPropertyOwned * final_price  # calculating total value
    roi = ((final_value - 1_000_000) / 1_000_000) * 100  # computing ROI

    # logging metrics every 100 episodes
    if (episode + 1) % 100 == 0:
        episode_rewards.append(total_reward)  # saving reward
        episode_rois.append(roi)  # saving ROI
        episode_values.append(final_value)  # saving portfolio value
        plotPoints.append(episode + 1)  # saving episode number
        print(f"Episode {episode+1}: Reward: ${total_reward:,.0f}, Portfolio: ${final_value:,.0f} ROI: {roi:.2f}%, Buy={buy}, Hold={hold}, Sell={sell}")


# plotting training performance

plt.figure(figsize=(12, 6))  # setting figure size
plt.plot(plotPoints, episode_values, marker='o')  # plotting portfolio values
plt.title('Portfolio Value Every 100 Episodes')  # setting title
plt.xlabel('Episode')  # labeling x-axis
plt.ylabel('Portfolio Value ($)')  # labeling y-axis
plt.grid(True)  # showing grid
plt.show()  # displaying plot

plt.figure(figsize=(12, 6))  # creating another figure
plt.plot(plotPoints, episode_rois, label='ROI (%)', marker='x', color='orange')  # plotting ROI
plt.title('ROI Trend Every 100 Episodes')  # setting title
plt.xlabel('Episode')  # labeling x-axis
plt.ylabel('Return on Investment (%)')  # labeling y-axis
plt.legend()  # showing legend
plt.grid(True)  # showing grid
plt.tight_layout()  # adjusting layout
plt.show()  # displaying plot
