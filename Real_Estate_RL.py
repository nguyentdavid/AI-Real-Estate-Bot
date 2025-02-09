# Importing required libraries
import gym  # For building reinforcement learning environments
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation and analysis
import random  # For generating random numbers (used in epsilon-greedy policy)
from sklearn.preprocessing import MinMaxScaler  # For normalizing data between 0 and 1
from gym import spaces  # For defining action and observation spaces in Gym
import matplotlib.pyplot as plt  # For plotting results at the end

# Preparing the CSV files to be read and simulate a real estate market environment
# Load the CSV files
real_estate_data = pd.read_csv('real_estate_data.csv', usecols=['DATE', 'HPI'])  # Load real estate HPI data
economic_indicators = pd.read_csv('economic_indicators.csv', usecols=['DATE', 'DIV', 'GDP'])  # Load economic indicators

# Renaming columns for easier read/manipulation
real_estate_data.rename(columns={'DATE': 'Date', 'HPI': 'HPI'}, inplace=True)
economic_indicators.rename(columns={'DATE': 'Date', 'DIV': 'DividendsInterestRent', 'GDP': 'GDP'}, inplace=True)

# Merge the datasets on 'Date' to align real estate and economic indicators data as there are 2 CSV files
data = real_estate_data.merge(economic_indicators, on='Date')  # Merging datasets on the Date column
data['Date'] = pd.to_datetime(data['Date'])  # Ensure Date is in datetime format for consistency
data = data.ffill()  # Padding any dates that are missing values

# Simulating house prices using a base year and the HPI
base_year = '2000-01-01'  # Define a base year for simulation of property prices
base_price = 166000  # median house price of Colorado in the year 2000
base_year_hpi = data.loc[data['Date'] == base_year, 'HPI'].values[0]  # Get HPI value for the base year

# Calculating simulated house prices for all rows based on HPI changes relative to the base year
data['PropertyPrice'] = (base_price * (data['HPI'] / base_year_hpi)).round().astype(int)

# Normalizing the data (except for 'Date') for input into the reinforcement learning model
scaler = MinMaxScaler()  # Initialize MinMaxScaler to normalize feature values
columns_to_scale = [col for col in data.columns if col not in ['Date', 'PropertyPrice']]
scaled_data = scaler.fit_transform(data[columns_to_scale])
data_scaled = pd.DataFrame(scaled_data, columns=columns_to_scale)
data_scaled['PropertyPrice'] = data['PropertyPrice']  # Add back the original unscaled 'PropertyPrice'


################################# Creating custom environment for agent ###############################################
# Creating the custom Real Estate environment class
class RealEstateEnv(gym.Env):
    def __init__(self, data):
        super(RealEstateEnv, self).__init__()  # Initializing parent class
        self.data = data  # Storing input data
        self.currentStep = 0  # Initializing current step (counter)
        self.balance = 1000000  # Starting balance
        self.numPropertyOwned = 0  # Number of properties held
        self.action_space = spaces.Discrete(3)  # Actions: 0 = Hold, 1 = Buy, 2 = Sell
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.data.shape[1] - 1,), dtype=np.float32  # Exclude PropertyPrice from observation
        )

    def reset(self):
        self.currentStep = 0  # Reset the step counter for every episode
        self.balance = 1000000  # Reset the balance to 1,000,000 for every episode
        self.numPropertyOwned = 0  # Reset properties held for every episode
        return self.data.drop(columns=['PropertyPrice']).iloc[self.currentStep].values  # Return initial state

    def step(self, action):
        currentPrice = self.data.iloc[self.currentStep]['PropertyPrice']  # Get the current property price

        reward = 0  # Initialize reward
        if action == 1:  # Buy property
            if self.balance >= currentPrice:  # Ensure enough balance
                self.numPropertyOwned += 1  # Increase properties held
                self.balance -= currentPrice  # Deduct property cost from balance
        elif action == 2:  # Sell property
            if self.numPropertyOwned > 0:  # Ensure agent HAS properties to sell
                self.numPropertyOwned -= 1  # Decrease properties held
                self.balance += currentPrice  # Add cost of property (when sold) to balance
                reward = currentPrice  # Assign reward based on sale price

        self.currentStep += 1  # next step
        done = self.currentStep >= len(self.data) - 1  # Checking if the simulation ends
        nextState = self.data.drop(columns=['PropertyPrice']).iloc[self.currentStep].values  # Get next state
        portfolioValue = self.balance + self.numPropertyOwned * self.data.iloc[self.currentStep][
            'PropertyPrice']  # Calculating portfolio value

        reward += portfolioValue - self.balance  # Adjusting the reward to encourage growth
        return nextState, round(reward, 3), done, {}  # Return state, reward, and done status

    def render(self, mode='human'):
        currentPrice = self.data.iloc[self.currentStep]['PropertyPrice']  # Get the current price
        print(f"Step: {self.currentStep}")  # Printing step count
        print(f"Balance: ${self.balance:,.2f}")  # Printing current balance
        print(f"Properties Held: {self.numPropertyOwned}")  # Printing properties held


################################# End of custom environment for agent ###############################################

#####################################################################################################################
# Q-learning Algorithm portion for training the agent using epsilon-greedy policy

# These can be adjusted at your discretion if you want to see different results with different parameters
# Q-learning parameters
alpha = 0.5  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate: I chose 1 because I wanted my agent to explore as much as possible
epsilon_decay = 0.99  # Decay rate for epsilon
epsilon_min = 0.05  # Minimum value for epsilon

q_table = np.zeros((len(data_scaled.columns), 3))  # Initialize Q-table

env = RealEstateEnv(data_scaled)  # Create environment instance

# Training loop for Q-learning
episode_rewards = []  # List to track total rewards
episode_portfolioValues = []  # List to track portfolio values
plotPoints = []  # List to track episodes for plotting

for episode in range(1000):  # Loop through episodes
    state = env.reset()  # Reset environment
    totalRewards = 0  # Initialize total reward

    buyCounter = 0  # Counter for buy actions
    holdCounter = 0  # Counter for hold actions
    sellCounter = 0  # Counter for selling actions

    for step in range(500):  # Limit to 500 steps per episode
        stateIndex = np.argmax(state)  # Get state index

        if random.uniform(0, 1) < epsilon:  # Less than epsilon = agent will explore (take random action)
            action = env.action_space.sample()  # Random action
        else:  # Exploitation, agent takes the best action
            action = np.argmax(q_table[stateIndex, :])  # Best action

            # Update action counters
            if action == 0:
                holdCounter += 1  # Increasing hold count by 1
            elif action == 1:
                buyCounter += 1  # Increasing buy count by 1
            elif action == 2:
                sellCounter += 1  # Increasing sell count by 1

        nextState, reward, done, _ = env.step(action)  # Take action
        nextStateIndex = np.argmax(nextState)  # Get next state index

        # Update Q-value
        q_table[stateIndex, action] += alpha * (
                reward + gamma * np.max(q_table[nextStateIndex, :]) - q_table[stateIndex, action]
        )

        state = nextState  # Update state
        totalRewards += reward  # Accumulate reward

        if done:  # Checking if episode is done
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)  # Decay epsilon

    finalPrice = env.data.iloc[env.currentStep]['PropertyPrice']  # Get final property price
    finalPortfolioValue = env.balance + env.numPropertyOwned * finalPrice  # Calculating final portfolio value

    if (episode + 1) % 100 == 0:  # Track every 100 episodes
        episode_portfolioValues.append(finalPortfolioValue)
        plotPoints.append(episode + 1)
        # Calculating ROI for this episode
        portfolioROI = ((finalPortfolioValue - 1000000) / 1000000) * 100
        print(
            f"Episode {episode + 1}: "
            f"Total Reward: {totalRewards:,.2f}, "
            f"Portfolio Value: ${finalPortfolioValue:,.2f}, "
            f"Balance: ${env.balance:,.2f}, "
            f"Properties Held: {env.numPropertyOwned},",
            f"Portfolio Return on Investment (ROI): {portfolioROI:.2f}%"
        )
        # Print action counts
        print(f"Actions in Episode {episode + 1}: Buy: {buyCounter}, Hold: {holdCounter}, Sell: {sellCounter}")

# End of Q learning algorithm section
#####################################################################################################################

##################################### Training and Evaluation Section ###############################################
# Evaluation loop
evalRewards = []  # List to track evaluation rewards
evalPortfolioValues = []  # List to track evaluation portfolio values

for i in range(100):  # Perform 100 evaluation runs
    state = env.reset()  # Reset environment
    totalRewards = 0  # Initialize reward
    done = False  # Reset done flag

    while not done:  # Until episode ends
        stateIndex = np.argmax(state)  # Get state index
        action = np.argmax(q_table[stateIndex, :])  # Choose best action
        state, reward, done, _ = env.step(action)  # Take action
        totalRewards += reward  # Accumulate reward

    finalPrice = env.data.iloc[env.currentStep]['PropertyPrice']  # Get final property price
    finalPortfolioValue = env.balance + env.numPropertyOwned * finalPrice  # Calculating portfolio value
    evalRewards.append(totalRewards)  # Append total reward
    evalPortfolioValues.append(finalPortfolioValue)  # Append portfolio value

# Calculating averages
averageReward = sum(evalRewards) / len(evalRewards)  # Average reward over evaluations
averagePortfolio = sum(evalPortfolioValues) / len(evalPortfolioValues)  # Average portfolio value
averageROI = ((averagePortfolio - 1000000) / 1000000) * 100  # Calculating average return on investment (ROI)

# Print final evaluation results
print("\n--- HIGHER PORTFOLIO ROI = BETTER PERFORMANCE --- ")
print(f"\n--- Final Results ---")  # Separator for final results
print(f"Average Evaluation Reward: {averageReward:,.2f}")  # Printing average reward
print(f"Average Portfolio Value: ${averagePortfolio:,.2f}")  # Printing average portfolio value
print(f"Average Portfolio ROI: {averageROI:.2f}%")  # Printing average ROI

# Plot the final portfolio values every 100 episodes
plt.figure(figsize=(12, 6))  # figure size
plt.plot(plotPoints, episode_portfolioValues, label='Portfolio Value (Every 100 Episodes)',
         marker='o')  # Plot data points with markers
plt.title('Portfolio Value per 100 Episodes', fontsize=16)  # plot title
plt.xlabel('Episode', fontsize=14)  # Label for the x-axis
plt.ylabel('Portfolio Value ($)', fontsize=14)  # Label for the y-axis
plt.grid(True, linestyle='--', alpha=0.7)  # Grid
plt.legend(fontsize=12)  # Legend for graph
plt.show()  # Accidentally deleted at the time of my recording

################################ End of Training and Evaluation Section ###############################################
