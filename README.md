# twitter_ML_project
It's a sentiment analysis system on tweets of  twitter


TWITTER SENTIMENT ANALYSIS PROJECT

We will be working on GOOGLE COLAB as it does not need to install each and very module independtly and manually instead just using import commands we can easily import required modules.
We will start by using google colab and creating an account on kaggle for taking dataset of twitter tweets for analysis.Firstly we installed kaggle on google colab and then we proceeded with uploading dataset from kaggle to the notebook. Then we continued with the importing twitter dataset in csv form. Now csv file is not easy to handle so we will import pandas file for smooth handling. Pandas is nothing more than the structured data frames which will store the data and help in further doing analysis on those data. Then we will import re module as The re.search() method takes a regular expression pattern and a string and searches for that pattern within the string. Then we will be futher using nltk(natural language toolkit) - It helps convert text into numbers, which the model can then easily work with. We will also import sklearn a very famous and used library. ALso we can't feed textual data and need to convert it into numerical data so we need to import TfidfVectorizer. Now train_test_split is also imported to convert our original data into training data and test data. STOPWORDS are such words which won't add any influential meaning like i,we,such,some,etc.

Now we will import data from csv format to pandas dataframe in twitter_data.We will be using functions to check and print the data and use isnull() for checking any missing values.We must check that there is equal distrbution of positive and negative tweets in machine learning.
twitter_data.replace({'target':(4,1),inplace=True}) we do inplace here to make changes in the original data.

def stemming(content):
  stemmed_content=re.sub('[^a-zA-Z]',' ',content)     ->this line removes all character except a-z and A-Z. all punctuation and all removed
  stemmed_content=stemmed_content.lower()             ->this lines converts all to lower case 
  stemmed_content=stemmed_content.split()             -> this line split all the words individually of the tweet
  stemmed_content=[port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]    ->removes the all stopwords in tweet
  stemmed_content=' '.join(stemmed_content)            ->after all things done then join the words of indivdual tweets
  return stemmed_content

After this we will apply stemming and it will take approx 50min as there are 16 lakh tweets or 1.6 million tweets.

Now taken two variables X and Y, where X=tweet and Y=target. test_size=0.2 means u are taking 20% of the data to test data and rest are train data for 80%.
In ML we need data in numerical form so we will use vectorizer for this converting purpose.

model=LogisticRegression(max_iter=1000) this 1000 shows max number of times model can go through the data.
Accuracy score on the training data: 0.81018984375 this shows if u are giving 100 tweets it can predit 81 tweets correctly and 19 tweets maybe incorrectly.
Accuracy score on the test data: 0.7780375 here the accuracy score of training data and test data is almost nearby so we can say that the model is performing well where if the test data has shown accuracy of 40% then it won't be a good model and this situation is referred as 'Overfitting'.

filename= 'trained_model.sav'
pickle.dump(model,open(filename,'wb'))
use it for saving the file and further downloading it.
We can test it with any test cases further also.
