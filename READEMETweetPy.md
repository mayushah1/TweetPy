
## News Mood
'''
In this assignment, you'll create a Python script to perform a sentiment analysis of the Twitter activity of various news oulets, and to present your findings visually.

Your final output should provide a visualized summary of the sentiments expressed in Tweets sent out by the following news organizations: __BBC, CBS, CNN, Fox, and New York times__.

![output_10_0.png](output_10_0.png)

![output_13_1.png](output_13_1.png)

The first plot will be and/or feature the following:

* Be a scatter plot of sentiments of the last __100__ tweets sent out by each news organization, ranging from -1.0 to 1.0, where a score of 0 expresses a neutral sentiment, -1 the most negative sentiment possible, and +1 the most positive sentiment possible.
* Each plot point will reflect the _compound_ sentiment of a tweet.
* Sort each plot point by its relative timestamp.

The second plot will be a bar plot visualizing the _overall_ sentiments of the last 100 tweets from each organization. For this plot, you will again aggregate the compound sentiments analyzed by VADER.

The tools of the trade you will need for your task as a data analyst include the following: tweepy, pandas, matplotlib, seaborn, textblob, and VADER.

Your final Jupyter notebook must:

* Pull last 100 tweets from each outlet.
* Perform a sentiment analysis with the compound, positive, neutral, and negative scoring for each tweet. 
* Pull into a DataFrame the tweet's source acount, its text, its date, and its compound, positive, neutral, and negative sentiment scores.
* Export the data in the DataFrame into a CSV file.
* Save PNG images for each plot.

As final considerations:

* Use the Matplotlib and Seaborn libraries.
* Include a written description of three observable trends based on the data. 
* Include proper labeling of your plots, including plot titles (with date of analysis) and axes labels.
* Include an exported markdown version of your Notebook called  `README.md` in your GitHub repository.  
'''




```python
# Dependencies
import tweepy
import json
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from config import consumer_key, consumer_secret, access_token, access_token_secret

# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
```


```python
# Target Search newa channel
target_user = ("@BBC","@CNN","@nytimes","@FoxNews", "CBSNews")

# Lists to hold results
compound_list = []
negi = []
posi = []
neut = []
users = []
numbers = []
dates = []
oldest_tweet = None
```


```python
for user in target_user:
   # Retrieve 100 most recent tweets -- specifying a max_id
    public_tweets = api.user_timeline(user, count=100, 
                               result_type="recent", max_id=oldest_tweet)
    counter = 1

    # Loop through all tweets
    for tweet in public_tweets:
        # Run Vader Analysis on each tweet
        results = analyzer.polarity_scores(tweet["text"])

        # Add each value to the appropriate array and reduce counter
        compound_list.append(results["compound"])
        negi.append(results["neg"])
        posi.append(results["pos"])
        neut.append(results["neu"])
        users.append(user)
        dates.append(tweet["created_at"])
        numbers.append(counter)
        counter +=1
news_df = pd.DataFrame({"Channel":users, "Date":dates,"Compound":compound_list,"Negative":negi, 
                        "Positive":posi,"Neutral":neut, "Tweet_Ago": numbers})
```


```python
#Check the result
news_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Channel</th>
      <th>Compound</th>
      <th>Date</th>
      <th>Negative</th>
      <th>Neutral</th>
      <th>Positive</th>
      <th>Tweet_Ago</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>@BBC</td>
      <td>0.0000</td>
      <td>Fri Mar 09 19:03:04 +0000 2018</td>
      <td>0.0</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>@BBC</td>
      <td>0.5719</td>
      <td>Fri Mar 09 18:00:10 +0000 2018</td>
      <td>0.0</td>
      <td>0.764</td>
      <td>0.236</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>@BBC</td>
      <td>0.0000</td>
      <td>Fri Mar 09 17:18:55 +0000 2018</td>
      <td>0.0</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>@BBC</td>
      <td>0.4939</td>
      <td>Fri Mar 09 17:00:10 +0000 2018</td>
      <td>0.0</td>
      <td>0.656</td>
      <td>0.344</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>@BBC</td>
      <td>0.3818</td>
      <td>Fri Mar 09 16:56:32 +0000 2018</td>
      <td>0.0</td>
      <td>0.874</td>
      <td>0.126</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
news_df.to_csv("News_Tweets.csv")
```


```python
#check number of tweets
len(users)
```




    500




```python
type(news_df)
```




    pandas.core.frame.DataFrame




```python
#group the result using pivot_table function
pivot_df = news_df.pivot_table(index=["Tweet_Ago"], columns=["Channel"], values=["Compound"])
pivot_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="5" halign="left">Compound</th>
    </tr>
    <tr>
      <th>Channel</th>
      <th>@BBC</th>
      <th>@CNN</th>
      <th>@FoxNews</th>
      <th>@nytimes</th>
      <th>CBSNews</th>
    </tr>
    <tr>
      <th>Tweet_Ago</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.0000</td>
      <td>-0.5267</td>
      <td>0.7269</td>
      <td>0.4404</td>
      <td>-0.7650</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.5719</td>
      <td>0.7269</td>
      <td>0.0000</td>
      <td>-0.6486</td>
      <td>-0.8555</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0000</td>
      <td>0.3612</td>
      <td>-0.2023</td>
      <td>-0.6463</td>
      <td>-0.3400</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.4939</td>
      <td>0.0000</td>
      <td>-0.6486</td>
      <td>0.0000</td>
      <td>-0.2960</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.3818</td>
      <td>-0.0258</td>
      <td>0.0000</td>
      <td>-0.1779</td>
      <td>-0.2960</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.5848</td>
      <td>0.5423</td>
      <td>-0.3400</td>
      <td>0.6369</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.4939</td>
      <td>0.7506</td>
      <td>-0.5423</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.2755</td>
      <td>-0.4215</td>
      <td>-0.6486</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.4939</td>
      <td>0.0000</td>
      <td>-0.5606</td>
      <td>-0.4939</td>
      <td>-0.2263</td>
    </tr>
    <tr>
      <th>10</th>
      <td>-0.4404</td>
      <td>-0.6486</td>
      <td>0.4404</td>
      <td>0.5574</td>
      <td>0.2960</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.7783</td>
      <td>0.0000</td>
      <td>0.0258</td>
      <td>0.0000</td>
      <td>-0.7269</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.0000</td>
      <td>-0.5423</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.2960</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.7579</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.5719</td>
      <td>0.4215</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.6114</td>
      <td>-0.7269</td>
      <td>0.7003</td>
      <td>0.0772</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.5859</td>
      <td>-0.3400</td>
      <td>-0.4939</td>
      <td>0.1139</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.9300</td>
      <td>0.0000</td>
      <td>-0.5267</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.6124</td>
      <td>0.4754</td>
      <td>-0.7650</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.4389</td>
      <td>-0.4552</td>
      <td>0.0000</td>
      <td>0.7096</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.0000</td>
      <td>0.2023</td>
      <td>-0.2960</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.4404</td>
      <td>-0.1027</td>
      <td>-0.4404</td>
      <td>0.4404</td>
      <td>-0.4939</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.0000</td>
      <td>-0.2960</td>
      <td>0.3818</td>
      <td>-0.0258</td>
      <td>-0.5574</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.4019</td>
      <td>0.7269</td>
      <td>0.0516</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.0000</td>
      <td>-0.2960</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.0000</td>
      <td>0.1280</td>
      <td>-0.6222</td>
      <td>-0.8603</td>
      <td>0.5106</td>
    </tr>
    <tr>
      <th>26</th>
      <td>-0.1027</td>
      <td>-0.4404</td>
      <td>0.1027</td>
      <td>-0.1531</td>
      <td>0.6801</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.0000</td>
      <td>-0.3400</td>
      <td>0.1027</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.5719</td>
      <td>-0.5994</td>
      <td>0.0000</td>
      <td>-0.6808</td>
      <td>0.4404</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.5803</td>
      <td>0.5106</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.7269</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0.0000</td>
      <td>-0.5574</td>
      <td>0.0000</td>
      <td>-0.1531</td>
      <td>0.7506</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>71</th>
      <td>0.4215</td>
      <td>0.3612</td>
      <td>-0.9042</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>72</th>
      <td>-0.6808</td>
      <td>0.3818</td>
      <td>-0.8271</td>
      <td>0.4047</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>73</th>
      <td>0.4215</td>
      <td>0.4588</td>
      <td>-0.5106</td>
      <td>0.5719</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>74</th>
      <td>-0.8591</td>
      <td>0.2263</td>
      <td>-0.3400</td>
      <td>0.0000</td>
      <td>-0.2960</td>
    </tr>
    <tr>
      <th>75</th>
      <td>0.5994</td>
      <td>-0.2263</td>
      <td>-0.5574</td>
      <td>0.0000</td>
      <td>0.8126</td>
    </tr>
    <tr>
      <th>76</th>
      <td>0.0000</td>
      <td>-0.3400</td>
      <td>-0.9100</td>
      <td>0.3400</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>77</th>
      <td>-0.4215</td>
      <td>-0.5423</td>
      <td>-0.2960</td>
      <td>-0.2748</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>78</th>
      <td>0.4939</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.1280</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>79</th>
      <td>-0.1027</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.6124</td>
    </tr>
    <tr>
      <th>80</th>
      <td>0.0000</td>
      <td>0.4549</td>
      <td>0.4927</td>
      <td>0.0000</td>
      <td>0.2023</td>
    </tr>
    <tr>
      <th>81</th>
      <td>0.3182</td>
      <td>0.2023</td>
      <td>-0.5574</td>
      <td>-0.7351</td>
      <td>-0.2263</td>
    </tr>
    <tr>
      <th>82</th>
      <td>-0.0644</td>
      <td>0.0000</td>
      <td>-0.2023</td>
      <td>0.0000</td>
      <td>-0.8402</td>
    </tr>
    <tr>
      <th>83</th>
      <td>0.6997</td>
      <td>-0.4404</td>
      <td>-0.2023</td>
      <td>0.4019</td>
      <td>0.4215</td>
    </tr>
    <tr>
      <th>84</th>
      <td>0.8439</td>
      <td>0.1027</td>
      <td>0.3612</td>
      <td>0.3400</td>
      <td>-0.5719</td>
    </tr>
    <tr>
      <th>85</th>
      <td>0.7269</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.7650</td>
      <td>0.3182</td>
    </tr>
    <tr>
      <th>86</th>
      <td>0.6246</td>
      <td>0.1027</td>
      <td>0.4215</td>
      <td>-0.2023</td>
      <td>0.3818</td>
    </tr>
    <tr>
      <th>87</th>
      <td>0.0000</td>
      <td>-0.3400</td>
      <td>0.0000</td>
      <td>0.2732</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>88</th>
      <td>0.4404</td>
      <td>0.0000</td>
      <td>0.7146</td>
      <td>0.0000</td>
      <td>0.3612</td>
    </tr>
    <tr>
      <th>89</th>
      <td>0.5994</td>
      <td>-0.4404</td>
      <td>0.7269</td>
      <td>-0.6486</td>
      <td>0.0258</td>
    </tr>
    <tr>
      <th>90</th>
      <td>0.7901</td>
      <td>0.3400</td>
      <td>0.0936</td>
      <td>0.0000</td>
      <td>-0.6486</td>
    </tr>
    <tr>
      <th>91</th>
      <td>0.4215</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>92</th>
      <td>-0.2732</td>
      <td>-0.4767</td>
      <td>-0.8271</td>
      <td>0.0000</td>
      <td>0.5106</td>
    </tr>
    <tr>
      <th>93</th>
      <td>0.5023</td>
      <td>-0.2960</td>
      <td>-0.5267</td>
      <td>0.2263</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>94</th>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.8402</td>
    </tr>
    <tr>
      <th>95</th>
      <td>-0.8555</td>
      <td>-0.8074</td>
      <td>0.7269</td>
      <td>0.7351</td>
      <td>-0.4767</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0.6124</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.5574</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>97</th>
      <td>0.4472</td>
      <td>0.4549</td>
      <td>0.4588</td>
      <td>0.0000</td>
      <td>-0.0424</td>
    </tr>
    <tr>
      <th>98</th>
      <td>0.1531</td>
      <td>0.2960</td>
      <td>0.7650</td>
      <td>0.0000</td>
      <td>-0.2732</td>
    </tr>
    <tr>
      <th>99</th>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.4215</td>
      <td>-0.3182</td>
    </tr>
    <tr>
      <th>100</th>
      <td>-0.1779</td>
      <td>0.7096</td>
      <td>-0.3818</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 5 columns</p>
</div>




```python
type(pivot_df)
```




    pandas.core.frame.DataFrame




```python
#Check column names for scatter plot
pivot_df.columns
```




    MultiIndex(levels=[['Compound'], ['@BBC', '@CNN', '@FoxNews', '@nytimes', 'CBSNews']],
               labels=[[0, 0, 0, 0, 0], [0, 1, 2, 3, 4]],
               names=[None, 'Channel'])




```python
# # Incorporate the other graph properties
now = datetime.now()
now = now.strftime("%Y-%m-%d %H:%M")
plt.title("Sentiment Analysis of Media Tweets ({})".format(now))
plt.ylabel("Tweet Polarity")
plt.xlabel("Tweets Ago")
#plot News channels
plt.scatter(np.arange(100),pivot_df['Compound']["@BBC"])
plt.scatter(np.arange(100),pivot_df['Compound']["@CNN"])
plt.scatter(np.arange(100),pivot_df['Compound']["CBSNews"])
plt.scatter(np.arange(100),pivot_df['Compound']["@FoxNews"])
plt.scatter(np.arange(100),pivot_df['Compound']["@nytimes"])
plt.legend(loc='best',bbox_to_anchor=(1,1))
#Save Plot
plt.savefig("TweetPy.png")
plt.show()
```


![png](output_12_0.png)



```python
group_df = news_df.groupby(["Channel"])[["Compound"]].mean()
```


```python
group_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound</th>
    </tr>
    <tr>
      <th>Channel</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>@BBC</th>
      <td>0.171544</td>
    </tr>
    <tr>
      <th>@CNN</th>
      <td>-0.055181</td>
    </tr>
    <tr>
      <th>@FoxNews</th>
      <td>-0.071107</td>
    </tr>
    <tr>
      <th>@nytimes</th>
      <td>0.039502</td>
    </tr>
    <tr>
      <th>CBSNews</th>
      <td>-0.052768</td>
    </tr>
  </tbody>
</table>
</div>




```python
# # Incorporate the other graph properties
plt.bar(1,group_df['Compound']["@BBC"])
plt.bar(2,group_df['Compound']["@CNN"])
plt.bar(3,group_df['Compound']["@FoxNews"])
plt.bar(4,group_df['Compound']["@nytimes"])
plt.bar(5,group_df['Compound']["CBSNews"])
plt.title("Overall Media Sentiments on Tweeter ({})".format(now))
plt.ylabel("Tweet Polarity")
plt.xlabel("Channels")
plt.ylim(-.2,.2)
plt.xticks(np.arange(6), (" ",'BBC','CNN','Fox News','NY Times','CBS'))
#Save Plot
plt.savefig("Media_TweetPy.png")
plt.show()
```


![png](output_15_0.png)

