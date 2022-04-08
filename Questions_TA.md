## 08-04 TA Session questions

- How much days can we use for the prediction od the next day? intuition says 1 week?
- As the lagestypart if the datas are from june, should we discard the rest? or maybe the everything from march and april?

- to what extend do we do statistical analyis?
- which statistics for the general analysis?

- ask about how we should approach the domain expert part of EDA?
- what is the difference between valence aand arousal?

Just take all months of data 

avergage per day

apps kan je weg laten, maar moet kunnen laten zien dat ze niet significant zijn dmv of wel data analyse of wel referentie



met lege waarden kijken wanneer ze zijn gerapporteerd. begin, eind of tussendoor

# data division:
- mood: final 
- circumplex [arousal, valence]
- countables [call, sms]
- time based [ appCat.all ]
- screen time 

## To Do's after session: 
- [ ] write func that sums call and sms data per hour/day
- [ ] write func that sums/means of the appcat data
- [ ] decide on what to do with the screen time
- [ ] which statistics for general statistics?
- [ ] look for outliers 
- [ ] look for errors
- [ ] proces NA vqalues for circumplex.valence
- [ ] look for missing values
- [ ] Data reduction!!!
- [ ]: decide on which variable we should apply what to do with these "structural inconsistencies"
- [ ]: determine which of these values are redundant aka analyse single variables piece by piece
- [ ]: Check for nan/corrupt values
- [ ]: Do some literature researche to which aggregation methods to use for which variables

# infrequent used apps:
finance, weather and game

check the ratio of app usage for only user that have the specific low usage app
-> not significant -> discardable
