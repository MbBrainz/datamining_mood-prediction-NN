## 08-04 TA Session questions

- How much days can we use for the prediction od the next day? intuition says 1 week?
- As the lagestypart if the datas are from june, should we discard the rest? or maybe the everything from march and april?


- should we check if the mood variable is really there for each hour of every data sample? or can we average the mood values of per day?
- should we sum the activity(appCat) per day or per hour ? 
- would averaging of activity per hour per day be of any use?
- ask about how we should approach the domain expert part of EDA?
- which statistics for the general analysis?



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