'''Name: Gordon Miller
   Title: Simple and Multiple Regression Sample
   Last Date Edited: July 28, 2023
'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:
tennis = pd.read_csv(r"C:\Users\12517\OneDrive - King's College\CodeCademy\Data Science\Python\Projects\tennis_ace_starting\tennis_stats.csv") #Replace your file path here
print(tennis.head())
print(tennis.columns)

# perform exploratory analysis here:
x = tennis[['Aces', 'DoubleFaults', 'FirstServe', 'FirstServePointsWon', 
            'SecondServePointsWon', 'BreakPointsFaced', 'BreakPointsSaved', 
            'ServiceGamesPlayed', 'ServiceGamesWon', 'TotalServicePointsWon',
            'FirstServeReturnPointsWon', 'SecondServeReturnPointsWon',
            'BreakPointsOpportunities', 'BreakPointsConverted',
            'ReturnGamesPlayed', 'ReturnGamesWon', 'ReturnPointsWon',
            'TotalPointsWon'
            ]]
y = tennis[['Wins', 'Losses', 'Winnings', 'Ranking']]

#plt.scatter(x, y)
#plt.xlabel('Aces')
#plt.ylabel('Wins')
#plt.show()
#plt.clf()

## scatter plots of each independent and dependent variable combination
for j in y:
   for i in x: 
       plt.figure()  # Create a new figure for each plot
       plt.scatter(x[i], y[j], alpha=0.5)  # Plot each scatterplot
       plt.xlabel(i)
       plt.ylabel(j)
       plt.title(j + ' vs ' + i)
       plt.show()
       plt.clf()

# perform a few single feature linear regressions here:

## Wins~Aces
x = tennis[['Aces']]
y = tennis[['Wins']]
    
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, 
                                                    test_size = 0.2)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

lm = LinearRegression()

model = lm.fit(x_train, y_train)
print(model.coef_, model.intercept_)

y_predict = lm.predict(x_test)

print("Train score (Wins ~ Aces): " + str(lm.score(x_train, y_train)))
print("Test score: (Wins ~ Aces" + str(lm.score(x_test, y_test)))

plt.scatter(y_test, y_predict, color='g', alpha=0.5)
plt.xlabel('Actual Wins')
plt.ylabel('Predicted Wins')
plt.title('Actual vs. Predicted Wins Using Aces (Wins ~ Aces)')
plt.show()
plt.clf()

# Wins ~ DoubleFaults

x = tennis[['DoubleFaults']]
y = tennis[['Wins']]
    
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, 
                                                    test_size = 0.2)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

lm = LinearRegression()

model = lm.fit(x_train, y_train)
print(model.coef_, model.intercept_)

y_predict = lm.predict(x_test)

print("Train score (Wins ~ DoubleFaults): " + str(lm.score(x_train, y_train)))
print("Test score (Wins ~ DoubleFaults): " + str(lm.score(x_test, y_test)))

plt.scatter(y_test, y_predict, color='g', alpha=0.5)
plt.xlabel('Actual Wins')
plt.ylabel('Predicted Wins')
plt.title('Actual vs. Predicted Wins Using DoubleFaults (Wins ~ DoubleFaults)')
plt.show()
plt.clf()

# Losses ~ ReturnGamesPlayed

x = tennis[['ReturnGamesPlayed']]
y = tennis[['Losses']]
    
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, 
                                                    test_size = 0.2)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

lm = LinearRegression()

model = lm.fit(x_train, y_train)
print(model.coef_, model.intercept_)

y_predict = lm.predict(x_test)

print("Train score (Losses ~ ReturnGamesPlayed): " + str(lm.score(x_train, y_train)))
print("Test score (Losses ~ ReturnGamesPlayed): " + str(lm.score(x_test, y_test)))

plt.scatter(y_test, y_predict, color='g', alpha=0.5)
plt.xlabel('Actual Losses')
plt.ylabel('Predicted Losses')
plt.title('Actual vs. Predicted Losses Using ReturnGamesPlayed (Losses ~ ReturnGamesPlayed')
plt.show()
plt.clf()


# perform a couple of two feature linear regressions here:

## Wins ~ Aces + ServiceGamesPlayed 

x = tennis[['Aces', 'ServiceGamesPlayed']]
y = tennis[['Wins']]
    
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, 
                                                    test_size = 0.2)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

lm = LinearRegression()

model = lm.fit(x_train, y_train)
print(model.coef_, model.intercept_)

y_predict = lm.predict(x_test)

print("Train score (Wins ~ Aces + ReturnGamesPlayed): " + str(lm.score(x_train, y_train)))
print("Test score (Wins ~ Aces + ReturnGamesPlayed): " + str(lm.score(x_test, y_test)))

plt.scatter(y_test, y_predict, color='g', alpha=0.5)
plt.xlabel('Actual Wins')
plt.ylabel('Predicted Wins')
plt.title('Actual Wins vs. Predicted Wins (Wins ~ Aces + ServiceGamesPlayed)')
plt.show()
plt.clf()

## Wins ~ DoubleFaults + ServiceGamesPlayed
    
x = tennis[['DoubleFaults', 'ServiceGamesPlayed']]
y = tennis[['Wins']]
    
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, 
                                                    test_size = 0.2)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

lm = LinearRegression()

model = lm.fit(x_train, y_train)
print(model.coef_, model.intercept_)

y_predict = lm.predict(x_test)

print("Train score (Wins ~ DoubleFaults + ServiceGamesPlayed): " + str(lm.score(x_train, y_train)))
print("Test score ~ DoubleFaults + ServiceGamesPlayed: " + str(lm.score(x_test, y_test)))

plt.scatter(y_test, y_predict, color='g', alpha=0.5)
plt.xlabel('Actual Wins')
plt.ylabel('Predicted Wins')
plt.title('Actual Wins vs. Predicted Wins (Wins ~ DoubleFaults + ServiceGamesPlayed)')
plt.show()
plt.clf()


# perform a multiple feature linear regression here:

## Winnings Model

x = tennis[['Aces', 'DoubleFaults', 'BreakPointsFaced', 'ServiceGamesPlayed',
            'BreakPointsOpportunities', 'ReturnGamesPlayed']]
y = tennis[['Winnings']]
    
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, 
                                                    test_size = 0.2)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

lm = LinearRegression()

model = lm.fit(x_train, y_train)
print('Coefficient (Winnings Model): '+ str(model.coef_))
print('Intercept (Winnings Model): ' + str(model.intercept_))

y_predict = lm.predict(x_test)

print("Train score (Winnings Model): " + str(lm.score(x_train, y_train)))
print("Test score (Winnings Model): " + str(lm.score(x_test, y_test)))

plt.scatter(y_test, y_predict, color='g', alpha=0.5)
plt.xlabel('Actual Wins')
plt.ylabel('Predicted Wins')
plt.title('Actual Wins vs. Predicted Wins (Winnings Model)')
plt.show()
plt.clf()