# Untitled_Trading_Bot

Something useful i hope

## Ideas
I need a comprehensive ML project - RL trading bot with visualization and explanations via LLM.
I need this project to get internship.

Small description: I want to build a system - full-stack web-application, where user can interact with some ui.
As the user i can: look at the current candlestick chart, try to predict the price, see the models prediction, get some explanation, why the models decided to predict it, get explanation of the strategy.
Main features:
- plot visualization
- synchronization with some market(s)
- Optimal trading strategy (via RL algorithms)
- LLM explanations

Main parts:
- Data collection and preprocessing pipeline (in on\off line formats) - i will train the model on some dataset (mostly collect by hands), but then, with some period of time, the model will be automatically re-trained on the most fresh data, which will be collect by the iterations with markets.
- RL algorithms - i will try to find and fine-tune some algorithm (most probable ppo) and also provide comparison with some others (like lstm, some rnns, or even with regression in terms of quality of predations).
- API and beautiful UI.
- Deployment, logging, and controlling of the model (most probable - mlflow)
- Production-ready code

## Done
First, i need to determine what i trade - usdt, gold, bitcoin, stocks. It seems to me that there is no great difference for me, since i am a ML engineer. However, maybe some of these examples can show very undeterministic behavior and undeterminable patterns, so all my attempts will be collapsed and i will cry. So i need to choose carefully: i need the good with the great amount of data and some predictable patterns.
Questions:
    1. What type of good should i choose?

## Partially Done
Second, i need to get the data for training. I need to find datasets on Kaggle or Google Datasets Search so far and so forth. But such data can be too outdated. I need to directly take data from Markets like Binance or whatever. But this data will be available in very very small size. But, i can get it from multiple places, so if i will at least mix some data from fresh datasets from Kaggle and my own, i will get a good data for training. Additionally, i will be able to add the new feature to my bot - synchronization with the latest data from some markets and online training for model.
Questions:
    1. Do you find my strategy of data collection optimal?
    2. Suggest markets with open api so i can get data from it.

## Partially Done
Third, i need to determine what data i need. It is the easiest part - i need to have something like this - timestamps, the current price, price direction (up/down), how much the price changed (within some period of time), number of sellers and buyers (if available).
Questions:
    1. Will it be sufficient for my case.

Then, i will have to determine the model. Most probably, i will choose ppo. I will also train dqn, (sac?), and lstm, (decision tree/forest/xgboost - ?) to compare RL with classic ML. Then, i will deploy my model on a server and will monitor it via mlflow.
Questions:
    1. I want to retrain the model, when i will get enough new data. Is it optimal strategy? I want to do it automatically.
    2. What RL algorithms do you suggest to try?

Then - API and UI. I will use FastAPI and JavaScript with HTML and CSS. It is the most uninteresting part, so i will skip it for now.

After that (if i will have time), i will integrate some LLM for the strategy explanation. How should i pre-train it? What data i should pass together with the prompt. Most probably, i need to pass the plans of RL model, but how can i determine them?

So, up for now, i need to know:
1. Python
2. RL algorithms
3. Ways of hyperparamters search
4. Numpy, pandas, tensorflow, keras
5. FastAPI, JavaScript, HTML, CSS
6. MLflow, Docker
7. Market API

How is my project? What suggestions do you have? What additional features do you suggest? What can be replaced or even removed?