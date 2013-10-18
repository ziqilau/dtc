nDTCM


----------------------------------------------USAGE-------------------------------------------------


nDTCM is used to organize the data in social streams. We believe the meaningful semantic features discovered by nDTCM can also benefit the representation of social streams.  
Using nDTCM as a building block can further analyze sophisticated structures/patterns in social streams. 


-------------------------------------------Input Data Format Example----------------------------------------------


15 5 0,1,2,3,4, 18 14:1 43:1 45:1 31:1 40:1 49:1 6:2 7:2 33:1 34:2 22:1 2:1 8:1 32:1 26:1 27:2 53:1 23:1

split by space, accordingly denoted as EPOCH #PARTICIPANTS PARTICIPANTS(split by commas) #DISTINCT_WORDS WORD:FREQUENCY

each data instance occupy one line



----------------------------------------------NOTE-------------------------------------------------


"utils.cpp" and "utils.h" is developed by Chong Wang at http://www.cs.cmu.edu/~chongw/resource.html.
