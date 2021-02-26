This project does not predict lottery numbers accurately, but it could be used as a basis for similar - less random tasks. 

2.5 thousand examples simply isn't enough to truly validate the algorithms viability against 1/>45,000,000 odds.

As of right now, there are evident issues with the algorithm. There is significant bias meaning that the last number predicted is nearly always the same, therefore I do plan on continuing to update the project. I will be implementing new models and testing new parameters in an attempt to understand which could work better for this kind of a problem. I will possibly look at implementing dummy data, and I will certainly amend the default values that the algorithm uses. Hence, this project is far from finished - I do however hope it's interesting, and perhaps useful to someone.

Usage is very simple through the command line UI.

To train the model type '5' in the prompt.

The purpose of the separate Web Scraping functions is to get the most up to date results, weekly, without causing duplicate data. The train_test_split function later took its duty of splitting the data into different sets. 
