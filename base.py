from siegeapi import Auth
import asyncio
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
import time
import numpy

# Dataframe for storing up to 24 hours of player data (This is to prevent API overload for overretrieved players)
df = pd.read_csv("dataFrame.csv")

f = open("token.txt", "r")
# Login keys: Replace with your own (Doesn't need to be Siege owning account)
UBISOFT_EMAIL = f.readline(1)
UBISOFT_PASSW = f.readline(1)

expiration = 100000000000000000000000

# Functions meant to check whether given mode and player combination has data already stored within an alotted time
def checkExists(players, mode):
    if (df.loc[(df['mode'] == mode) & (df['usr'] == players) & (df['timeRecorded'] > time.time() - expiration)].shape[0] > 0):
        return True
    return False

# Function meant to check whether given mode and player combination has data already stored
def alreadyExists(players, mode):
    if (df.loc[(df['mode'] == mode) & (df['usr'] == players)].shape[0] > 0):
        return True
    return False


# This is meant for testing data retrieval (Default example offered with API)
async def sample(usr):
    auth = Auth(UBISOFT_EMAIL, UBISOFT_PASSW)
    player = await auth.get_player(name=usr)

    print(f"Name: {player.name}")
    print(f"Profile pic URL: {player.profile_pic_url}")

    await player.load_playtime()
    print(f"Total Time Played: {player.total_time_played:,} seconds / {player.total_time_played_hours:,} hours")
    print(f"Level: {player.level}")

    await player.load_ranked_v2()
    print(f"Ranked Points: {player.ranked_profile.rank_points}")
    print(f"Rank: {player.ranked_profile.rank}")
    print(f"Max Rank Points: {player.ranked_profile.max_rank_points}")
    print(f"Max Rank: {player.ranked_profile.max_rank}")

    await player.load_progress()
    print(f"XP: {player.xp:,}")
    print(f"Total XP: {player.total_xp:,}")
    print(f"XP to level up: {player.xp_to_level_up:,}")

    await auth.close()

# Creates dataframe corresponding to the player
async def createData(usrs, mode):
    global df
    auth = Auth(UBISOFT_EMAIL, UBISOFT_PASSW)
    output = pd.DataFrame(columns=["top_position", "time", "wins", "losses", "abandons", "kills", "deaths"])
    top_rank_position = 0
    # Three modes are needed as each has different playstyle (Quick Play might be less serious, while Ranked is more serious, leads to different outcomes)
    if mode == 0:
        for players in usrs:
            if checkExists(players, 0):
                # Grab latest profile
                playerStats = df.loc[(df['usr'] == players) & (df['mode'] == 0)].head(1)
                output.loc[len(output)] = [playerStats['top_rank_position'],  playerStats['pvp_time_played'],  playerStats['wins'],  playerStats['losses'],  playerStats['abandons'],  playerStats['kills'],  playerStats['deaths']]
                continue
            
            dt = time.time()
            player = await auth.get_player(name=players)
            id = player.id
            await player.load_playtime()
            pvp_time_played = player.pvp_time_played
            await player.load_ranked_v2()
            profile=player.casual_profile    
            
            # If profile doesn't exist, we will average it at the end
            if(profile is None):
                profile = player.unranked_profile
                if(profile is None):
                    profile = player.ranked_profile
                    if(profile is None):   
                        wins = float('NaN')
                        losses = float('NaN')
                        abandons = float('NaN')
                        kills = float('NaN')
                        deaths = float('NaN')
                    else:
                        wins = profile.wins
                        losses = profile.losses
                        abandons = profile.abandons
                        kills = profile.kills
                        deaths = profile.deaths                       
                else:
                    wins = profile.wins
                    losses = profile.losses
                    abandons = profile.abandons
                    kills = profile.kills
                    deaths = profile.deaths
            else:
                wins = profile.wins
                losses = profile.losses
                abandons = profile.abandons
                kills = profile.kills
                deaths = profile.deaths
            
            # Locate player at the end of output table
            output.loc[len(output)] = [top_rank_position, pvp_time_played, wins, losses, abandons, kills, deaths]
            
            # Creates new profile
            df.loc[len(df)] = [mode, players, id, top_rank_position, pvp_time_played, wins, losses, abandons, kills, deaths, dt]
    elif mode == 1:
        for players in usrs:
            if checkExists(players, 1):
                # Grab latest profile
                playerStats = df.loc[(df['usr'] == players) & (df['mode'] == 1)].head(1)
                output.loc[len(output)] = [playerStats['top_rank_position'],  playerStats['pvp_time_played'],  playerStats['wins'],  playerStats['losses'],  playerStats['abandons'],  playerStats['kills'],  playerStats['deaths']]
                continue
            if checkExists(players, 2):
                # Grab latest profile
                playerStats = df.loc[(df['usr'] == players) & (df['mode'] == 1)].head(1)
                output.loc[len(output)] = [playerStats['top_rank_position'],  playerStats['pvp_time_played'],  playerStats['wins'],  playerStats['losses'],  playerStats['abandons'],  playerStats['kills'],  playerStats['deaths']]
                continue
                        
            dt = time.time()
            player = await auth.get_player(name=players)
            id = player.id
            await player.load_playtime()
            pvp_time_played = player.pvp_time_played
            await player.load_ranked_v2()
            profile=player.unranked_profile 
               
            if(profile is None):
                profile = player.ranked_profile
                if(profile is None):
                    profile = player.casual_profile
                    if(profile is None):   
                        wins = float('NaN')
                        losses = float('NaN')
                        abandons = float('NaN')
                        kills = float('NaN')
                        deaths = float('NaN')
                    else:
                        wins = profile.wins
                        losses = profile.losses
                        abandons = profile.abandons
                        kills = profile.kills
                        deaths = profile.deaths                       
                else:
                    wins = profile.wins
                    losses = profile.losses
                    abandons = profile.abandons
                    kills = profile.kills
                    deaths = profile.deaths
            else:
                wins = profile.wins
                losses = profile.losses
                abandons = profile.abandons
                kills = profile.kills
                deaths = profile.deaths
            output.loc[len(output)] = [top_rank_position, pvp_time_played, wins, losses, abandons, kills, deaths]

            df.loc[len(df)] = [mode, players, id, top_rank_position, pvp_time_played, wins, losses, abandons, kills, deaths, dt]
    elif mode == 2:
        for players in usrs:
            if checkExists(players, 2):
                # Grab latest profile
                playerStats = df.loc[(df['usr'] == players) & (df['mode'] == 2)].head(1)
                output.loc[len(output)] = [playerStats['top_rank_position'],  playerStats['pvp_time_played'],  playerStats['wins'],  playerStats['losses'],  playerStats['abandons'],  playerStats['kills'],  playerStats['deaths']]
                continue
            if checkExists(players, 1):
                # Grab latest profile
                playerStats = df.loc[(df['usr'] == players) & (df['mode'] == 1)].head(1)
                output.loc[len(output)] = [playerStats['top_rank_position'],  playerStats['pvp_time_played'],  playerStats['wins'],  playerStats['losses'],  playerStats['abandons'],  playerStats['kills'],  playerStats['deaths']]
                continue
            
            dt = time.time()
            player = await auth.get_player(name=players)
            id = player.id
            await player.load_playtime()
            pvp_time_played = player.pvp_time_played
            
            await player.load_ranked_v2()
            profile= player.ranked_profile
           
            if(profile is None):
                profile = player.unranked_profile
                if(profile is None):
                    profile = player.casual_profile
                    if(profile is None):   
                        wins = float('NaN')
                        losses = float('NaN')
                        abandons = float('NaN')
                        kills = float('NaN')
                        deaths = float('NaN')
                    else:
                        wins = profile.wins
                        losses = profile.losses
                        abandons = profile.abandons
                        kills = profile.kills
                        deaths = profile.deaths                       
                else:
                    wins = profile.wins
                    losses = profile.losses
                    abandons = profile.abandons
                    kills = profile.kills
                    deaths = profile.deaths
            else:
                wins = profile.wins
                losses = profile.losses
                abandons = profile.abandons
                kills = profile.kills
                deaths = profile.deaths

            output.loc[len(output)] = [top_rank_position, pvp_time_played, wins, losses, abandons, kills, deaths]

            df.loc[len(df)] = [mode, players, id, top_rank_position, pvp_time_played, wins, losses, abandons, kills, deaths, dt]
    
    output = output.apply(lambda x: x.astype(float, errors='ignore'))
    
    output = output.apply(lambda x: x.fillna(x.mean()), axis=0) 
    
    output = (output - output.mean(numeric_only=True)) / output.std(numeric_only=True)
    
    output = output.fillna(0)

    #print(output.values)
    
    df = df.loc[df['timeRecorded'] > time.time() - expiration]
    df.to_csv("dataFrame.csv", encoding='utf-8', index=False)
    
    await auth.close()
    return output

# Custom data loader that converts dataframe to tensor
class teamDataset(Dataset):
    def __init__(self, dataframes, winner):
        self.teams=dataframes
        self.winners=winner
        
    def __getitem__(self, index):
        dataframe = torch.tensor(self.teams[index].to_numpy()).float()
        winner = torch.tensor(self.winners[index]).float()
        return dataframe, winner
        
    def __len__(self):
        return len(self.teams)

# Converts the training teams to usable dataframe and output pair
def convertToDataset(dataframe):
    input = []
    output = []
    for index, row in dataframe.iterrows():
        toAppend = asyncio.run(createData([row['player0'],row['player1'], row['player2'], row['player3'], row['player4'], row['player5'], row['player6'], row['player7'], row['player8'], row['player9']], row['mode']))
        input.append(toAppend)
        output.append(row['winner'])
        
    return teamDataset(dataframes=input, winner=output)

# Two layered neural net, input->hidden->output
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.flatten(x)
        logits = self.linear_relu_stack(y)
        return logits
    
    
class NNNonFlatten(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NNNonFlatten, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits