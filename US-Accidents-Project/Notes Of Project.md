- Selection of Columns
Following are the columns which I choose to not process for the below mentioned reasons
1) ID: Unique identifier
2) Number: More than half of the entries are null, so it's not useful to take street numbers into account
3) Country: All of the data is for US, so this entry doesn't add any information to the data.
4) Description: It's a text summary, which is useful for NLP, however, will take a look into it later, maybe wrt sentiment analysis
5) Turning Loop: All of the values are false so are of no use.

*6) Precipitation: The precipitation is 0 mostly, for most of the accident, but its good to take a look what is causing the accident if no rain.
*7) End time: Not so good idea, cause many unique values and not conveying more information about the accident. (We can see there will be a relationship between severity and diff. in Start time - End time )


Tranformation:
- Convert All day, night to binary values
- Add a combined feature of start-end to get total accident time
- Convert R,L in side of accident
- If the cardinality is <= 10 for the columns then one hot encode the columns..., also incldue, Wind directions 
- Take difference of latitudes and longitudes to get simple vertical and horizontal distances between accidents

-------------
Think what to do about street, city, County, State, Zipcode, Airport code, Weather timestamps, weather_condition
Obj: Street, City, County, State, Zipcode, Weather Timestamps, Weather Conditon
-------------

Think how to handle the categorical null values first and then the numerical null values.



# Predicting the severity of an accident


----------------------
### Processing the object data columns
- Convert Start and End time to proper format, and get it in form of month, year and day
- Calculate duration of accident in mins
- Remove description can be used for NLP
- Remove weather time-stamp since it's just another time stamp from weather station
- Keep zipcode, street, city, county and state **Think about how to handle these text values**
- Remove AirportCode as not much of help here, cause above bits give good info about spatial info.
- Weather Condition : Think about it's visualization
- Wind Direction: Think about it's viz
- All objective columns having cardinality < 10 to be hot encoded
- All day n night to be converted to binary values
- Convert side to again binary values
- Country to be removed

-----------------------
### Processing the numerical data columns
- Keep start_lng, start_lat, take differene between lng and lat to get orientation of accident
- Find the correlation between columns to the target value, and those which don't have a correlation drop those
- Drop columns with high number of missing values and those with a constant value for every instance
- Observe correlation between every numeric feature
-----------------------
- Use mode for imputation for float missing values, 
- Use most occurring value for object missing values

-------------------------------------------------------------------
