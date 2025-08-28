# Siege-Match-Predictor
Setup:
Install the latest version of nodejs and npm
Install React by running:
```
npm install react react-dom
```
Create a new Python virtual environment and then run: 
```
pip install -r requirements.txt
pip install -e .
npm ci .
```

You'll also need to create var/token.txt, and put in your Ubisoft email and password, one line after another

Finally, run python training/train.py to start the training process

To start the the web server, run:

```
flask --app MatchPredictorServer --debug run --host 0.0.0.0 --port 8000
```

Input format to predict: 
```
python utilize.py PREDICT <team A player 1> <team A player 2> <team A player 3> <team A player 4> <team A player 5> <team B player 1> <team B player 2> <team B player 3> <team B player 4> <team B player 5>
```
Input format to quit: 
```
python utilize.py QUIT
```
