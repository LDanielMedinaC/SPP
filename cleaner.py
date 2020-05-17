import pandas as pn 

happinessRank = pn.read_csv("happiness.csv")
population = pn.read_csv("population.csv")

happinessRank = happinessRank[["Country","Happiness Rank"]]
population = population[["Country", "Year_2016"]]

data = pn.merge(happinessRank, population, left_on = 'Country', right_on = 'Country')

data.to_csv("data.csv")

print(data)
