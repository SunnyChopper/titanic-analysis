# Analysis of Titanic data 
This is considered a classic machine learning problem. It's pretty much the equivalent of "Hello World" in machine learning. 

As you can tell, I'm just starting out on GitHub, so this is my "Hello World" into the machine learning world. Alright, enough of that, let me explain what I did and my thought process. 

So I was given a .csv file of 1,313 people who were on the Titanic when it crashed and it had multiple columns of data. Columns are called "features" in machine learning, so I will be referring to columns of data as features. So there were five main features. They were:

<ol>
  <li>Name</li>
  <li>Passenger Class</li>
  <li>Age</li>
  <li>Gender</li>
  <li>Survived</li>
</ol>

I chose my independent variables to be "passenger class", "age", and "gender". My dependent variable was "survived". 

I imported the data using the pandas library and then separated my independent variables by using the .iloc function. 

Next, I noticed that there was missing data in the dataset, so to take care of this, I used sklearn.preprocessing to import the Imputer object and replaced all missing data by the average of that feature. 

Then, I noticed I had categorical data, specifically, the passenger class feature was categorical, so I encoded this data. 

Next, I split the data into a training set and test set. 

Finally, I applied feature scaling to both the training set independent variables and the test set independent variables. In hindsight, I should have applied feature scaling before splitting the data. 

Then the magic happens when I apply the random forest classifier. The reason I picked the random forest classifier is because it very much depended on a few variables whether they let you on the rescue boats or not. For example, if your age was less than some number 'x' and you were a female, you were let on the boat. 

That example itself sounds like an if-then-else statement and random forests are good for that and since they are non-linear, they can pick up on the non-linear patterns. 

Finally, I created a small function that would let me know the reliability of my model based on the test set. I did not use the confusion matrix because I wanted to see if I can write my own function for this. 

Well, that's my "hello world" to the machine learning world and I hope to achieve greater things down the road. Thank you. 
