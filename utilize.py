import asyncio
import numpy as np
import torch
from torch import nn
from math import exp

from base import createData
from base import NNNonFlatten




device = (    
    "cuda"
   if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

best_acc = -np.inf

model = NNNonFlatten(70, 30).to(device)

model.load_state_dict(torch.load("model.pth", weights_only=True))

model.eval()

def predictTest(team, mode):
    with torch.no_grad():
        dataframe = asyncio.run(createData(team, mode))
        dataframe = dataframe.astype(float)
        dataframe = (dataframe - dataframe.mean())/dataframe.std()
        dataframe.fillna(0, inplace=True)
        x = torch.tensor(dataframe.values).float()
        # Needed due to evaluating model causing issues for some odd reason
        #x=torch.flatten(x)
        x = x.to(device)
        
        # Idk why flatten doesn't work within and must be done outside
        x = torch.flatten(x)
        pred = model(x)
        # First value typically is able to predict which side wins, with 0 meaning Team 1 and 1 meaning Team 2
        predicted = pred[0]
        print(f'Predicted Team: "{round(predicted.item()) + 1}"')
        
        # Value corresponds to percentage chance given team wins
        if(round(predicted.item()) == 1):
            print(f'Percentage chance team wins: "{(exp(predicted)/(1+exp(predicted)) * 100):>0.1f}"%')
        elif(round(predicted.item()) == 0):
            print(f'Percentage chance team wins: "{((1 - exp(predicted)/(1+exp(predicted))) * 100):>0.1f}"%')


def predict(team, mode):
    with torch.no_grad():
        dataframe = asyncio.run(createData(team, mode))
        dataframe = dataframe.astype(float)
        dataframe = (dataframe - dataframe.mean())/dataframe.std()
        dataframe.fillna(0, inplace=True)
        x = torch.tensor(dataframe.values).float()
        # Needed due to evaluating model causing issues for some odd reason
        #x=torch.flatten(x)
        x = x.to(device)
        
        # Idk why flatten doesn't work within and must be done outside
        x = torch.flatten(x)
        pred = model(x)
        # First value typically is able to predict which side wins, with 0 meaning Team 1 and 1 meaning Team 2
        predicted = pred[0]
        print(f'Predicted Team: "{round(predicted.item()) + 1}"')
        
        # Value corresponds to percentage chance given team wins
        if(round(predicted.item()) == 1):
            return 2, (exp(predicted)/(1+exp(predicted)) * 100)
        elif(round(predicted.item()) == 0):
            return 1, 1 - exp(predicted)/(1+exp(predicted)) * 100


test = False

# Basic command line for testing, this will be replaced with website link in
while test:
    command = input("Type in command:")
    if(command == "QUIT"):
        quit()
    
    if(command.split(' ', 1)[0] == "PREDICT"):
        splitted = command.split(' ')
        splitted.pop(0)
        mode = int(splitted[-1])
        splitted.pop(-1)
        predictTest(splitted, mode)
        
        
        
from flask import Flask, render_template, redirect, url_for,request
from flask import make_response
app = Flask(__name__)


@app.route('/')
def index():
  return render_template('mainWeb.html')

if __name__ == "__main__":
  app.run(debug=True)
  
@app.route('/', methods=['GET', 'POST'])
def index():
  teams = 1
  percent = 50
  if request.method == 'POST':
      form = request.form
      teams, percent = predict({form.tAone, form.tAtwo, form.tAthree, form.tAfour, form.tAfive, form.tBone, form.tBtwo, form.tBthree, form.tBfour, form.tBfive}, form.mode)
  return render_template('mainWeb.html', team=teams, percent=percent)
      