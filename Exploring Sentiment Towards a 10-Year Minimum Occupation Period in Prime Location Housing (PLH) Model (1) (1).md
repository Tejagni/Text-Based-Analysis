## Exploring Sentiment Towards a 10-Year Minimum Occupation Period in Prime Location Housing (PLH) Model

On October 27th, the Ministry of National Development (MND) and the Housing Development Board (HDB) announced the new Prime Location Housing (PLH) model. This is the first model of its kind in Singapore and aims to help lower to middle-income families live in the city area. Before releasing this model, they gathered public feedback and carefully planned it.

To understand how people feel about the new PLH model, we will do the following:

Collect data from web scraping using various tools and techniques. Process and clean the data (e.g., clean text, extract and convert emojis). Conduct sentiment analysis using TextBlob. Perform NLP analysis using NLTK (tokenization, POS-tagging, stemming, and removing stop words). Before starting data collection and cleaning, we expect most comments to be positive because of the extensive public engagement and feedback that shaped the model.

However, it's important to note that the data from web scraping is not a proper sample size and doesn't represent the entire Singapore population. We cannot get information about users' age, gender, marital status, or financial status from web scraping, so these factors are not included in the analysis.

## 1. Loading Packages


```python
# import praw
# from praw.models import MoreComments
import datetime
import pandas as pd
import numpy as np
import re
import pprint
pd.options.mode.chained_assignment = None

from textblob import TextBlob

import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import tokenize
import spacy
#conda install -c conda-forge spacy-model-en_core_web_sm
import en_core_web_sm
nlp = en_core_web_sm.load()
nltk.download('averaged_perceptron_tagger')

import matplotlib.pyplot as plt 
from matplotlib.pyplot import figure
import seaborn as sns

import emoji
```

    [nltk_data] Downloading package punkt to
    [nltk_data]     /Users/puttasathvik/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package wordnet to
    [nltk_data]     /Users/puttasathvik/nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!
    [nltk_data] Downloading package averaged_perceptron_tagger to
    [nltk_data]     /Users/puttasathvik/nltk_data...
    [nltk_data]   Package averaged_perceptron_tagger is already up-to-
    [nltk_data]       date!
    

## 2.Collecting and Organizing Comments and User Details Using Web Scraping

We'll collect all top-level (parent) comments and their replies using web scraping techniques. To keep track of which replies belong to which comments, we'll also collect the comment IDs, parent IDs, and usernames of the people who posted them. The usernames are important for the following reasons:

Deleted accounts with removed comments
Deleted accounts with unremoved comments
Bot accounts like RemindMeBot and sneakpeek_bot
When we clean the data, we'll need to remove these unwanted accounts while keeping the comments intact.

Note that comment.author returns an object. To get just the username, we need to use comment.author.name. We must check that comment.author is not None (to avoid deleted accounts), but since deleted accounts still have comments, we'll continue using comment.author.

Additionally, we'll store comments and user details separately.


```python
pip install requests beautifulsoup4
```

    Requirement already satisfied: requests in c:\users\tejag\anaconda3\lib\site-packages (2.31.0)
    Requirement already satisfied: beautifulsoup4 in c:\users\tejag\anaconda3\lib\site-packages (4.12.2)
    Requirement already satisfied: charset-normalizer<4,>=2 in c:\users\tejag\anaconda3\lib\site-packages (from requests) (2.0.4)
    Requirement already satisfied: idna<4,>=2.5 in c:\users\tejag\anaconda3\lib\site-packages (from requests) (3.4)
    Requirement already satisfied: urllib3<3,>=1.21.1 in c:\users\tejag\anaconda3\lib\site-packages (from requests) (1.26.16)
    Requirement already satisfied: certifi>=2017.4.17 in c:\users\tejag\anaconda3\lib\site-packages (from requests) (2023.11.17)
    Requirement already satisfied: soupsieve>1.2 in c:\users\tejag\anaconda3\lib\site-packages (from beautifulsoup4) (2.4)
    Note: you may need to restart the kernel to use updated packages.
    


```python
import requests
from bs4 import BeautifulSoup
import csv

# List of URLs to scrape
urls = [
    "https://www.kaggle.com/code/jyingong/data-cleaning-sentiment-analysis-reddit",
    "https://www.bleubricks.com/plh-prime-location-public-housing-what-are-the-ripple-effects-of-these-new-policies-for-the-hdb-market-and-beyond/",
    "https://www.theorigins.com.sg/post/7-pitfalls-of-the-new-10-year-mop-under-plh-model-you-should-know",
    "https://blog.seedly.sg/prime-location-public-housing-plh-hdb-flats/",
    "https://ifonlysingaporeans.blogspot.com/2021/10/hdb-prime-location-public-housing-plh.html"
]

# Function to scrape a single URL
def scrape_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract title
        title = soup.find('title').get_text() if soup.find('title') else 'No title found'
        # Extract the main content
        paragraphs = soup.find_all('p')
        content = '\n'.join([para.get_text() for para in paragraphs])
        return {
            'url': url,
            'title': title,
            'content': content
        }
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

# List to hold scraped data
data = []

# Scrape each URL and store the results in the data list
for url in urls:
    result = scrape_url(url)
    if result:
        data.append(result)

# Define the CSV file name
csv_file = 'scraped_data.csv'

# Write the data to the CSV file
with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=['url', 'title', 'content'])
    writer.writeheader()
    for row in data:
        writer.writerow(row)

print(f"Data has been written to {csv_file}")
```

    Data has been written to scraped_data.csv
    


```python
import pandas as pd
df = pd.read_csv("scraped_data.csv")
```

## 3. Data Cleaning and Processing 

The data is stored in two separate files: df_comments and df_details. We'll focus on cleaning the comments in df_comments. It's important to make sure that the number of rows stays the same before and after cleaning, so no data is lost.

Here are the steps to clean the data:

Remove links and link markups.
Remove double spaces that appear as "​".
Remove irrelevant comments (like deleted comments or those made by bots).
Remove duplicate comments (such as parent comments repeated in replies, like in WhatsApp).
Clean up text (fix contractions, typos, slang, emojis, etc.).

## 3a. Removing Duplicate Comments (parent comments in replies), Links, and Link Markups

For duplicate comments, the PRAW API doesn't remove parent comments in replies. To fix this, we need to split the comments using "\n\n" as a separator and then expand the nested list into individual rows. Duplicate parent comments start with ">", so we can replace these comments with an empty string ("") to keep the number of rows the same.

Here's an example:

">It will be located on two plots of land along Weld Road and Kelantan Road next to Jalan Besar MRT station, said the National Development Ministry and Housing Board in a joint statement on Wednesday (Oct 27).\n\nok so now we know they took away the rent-free Sungei Road flea market in 2017 and the open space carpark beside SLT for this. \n\n​\n\n>Besides an MRT station at their doorstep, future residents will also be within walking distance of Berseh Food Centre and Stamford Primary School"

After cleaning and processing, we can group and join the rows back together.









```python
#check number of rows
len(df_comment)
```




    358




```python
#check for for any empty cells 
df_comment.isna().sum()
```




    comment    0
    dtype: int64




```python
#split sentences by delimited "\n\n"
df_comment["comment"] = [item.split("\n\n") for item in df_comment.comment]

#explode nested list into individual rows 
df_comment = df_comment.explode("comment").rename_axis("index_name").reset_index()

#replace double space with empty string
df_comment["comment"] = df_comment.comment.str.replace("&#x200B;", "")
```


```python
#check number of rows after exploding
len(df_comment)
```




    580




```python
#for replies with parent comments within, remove parent comment and retain replies  
#those are fields with string that start with ">"
#remove bullet points 
df_comment.loc[df_comment.comment.str.startswith(">")] = "/Users/puttasathvik/Downloads/TextBasedAnalysis/Final Project/comment.csv"
df_comment["comment"] = [i.strip() for i in df_comment.comment]
df_comment["comment"] = [re.sub(r"^[0-9]", " ", i) for i in df_comment.comment]
```

    /var/folders/5g/277wtpf90l59q6v17xfw76h00000gn/T/ipykernel_37063/749234558.py:4: FutureWarning: Setting an item of incompatible dtype is deprecated and will raise in a future error of pandas. Value '/Users/puttasathvik/Downloads/TextBasedAnalysis/Final Project/comment.csv' has dtype incompatible with int64, please explicitly cast to a compatible dtype first.
      df_comment.loc[df_comment.comment.str.startswith(">")] = "/Users/puttasathvik/Downloads/TextBasedAnalysis/Final Project/comment.csv"
    


```python
#see table of items with https links and markup links 
df_comment.loc[df_comment.comment.str.contains("https")]
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
      <th>index_name</th>
      <th>comment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>36</th>
      <td>19</td>
      <td>The existing Stamford Primary School nearby ha...</td>
    </tr>
    <tr>
      <th>56</th>
      <td>20</td>
      <td>---\n1.0.2 | [Source code](https://github.com/...</td>
    </tr>
    <tr>
      <th>336</th>
      <td>219</td>
      <td>Yup - or they can decide not to get married [l...</td>
    </tr>
    <tr>
      <th>503</th>
      <td>322</td>
      <td>https://www.ons.gov.uk/peoplepopulationandcomm...</td>
    </tr>
    <tr>
      <th>509</th>
      <td>325</td>
      <td>https://www.iras.gov.sg/taxes/individual-incom...</td>
    </tr>
    <tr>
      <th>516</th>
      <td>329</td>
      <td>I will be messaging you in 11 years on [**2032...</td>
    </tr>
    <tr>
      <th>517</th>
      <td>329</td>
      <td>[**CLICK THIS LINK**](https://www.reddit.com/m...</td>
    </tr>
    <tr>
      <th>518</th>
      <td>329</td>
      <td>^(Parent commenter can ) [^(delete this messag...</td>
    </tr>
    <tr>
      <th>520</th>
      <td>329</td>
      <td>|[^(Info)](https://www.reddit.com/r/RemindMeBo...</td>
    </tr>
    <tr>
      <th>529</th>
      <td>335</td>
      <td>https://www.reddit.com/r/singapore/comments/8n...</td>
    </tr>
    <tr>
      <th>560</th>
      <td>350</td>
      <td>you [said](https://old.reddit.com/r/singapore/...</td>
    </tr>
  </tbody>
</table>
</div>




```python
#do a temporary table to see the usernames for these comments 

df_temp_https = df_comment[df_comment.comment.str.contains("https")]
df_temp_details = df_details.reset_index()
df_temp = pd.merge(df_temp_https, df_temp_details, how = "inner", left_on = "index_name", right_on = "index")
df_temp
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
      <th>index_name</th>
      <th>comment</th>
      <th>index</th>
      <th>comment_id</th>
      <th>parent_id</th>
      <th>username</th>
      <th>upvotes</th>
      <th>datetime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19</td>
      <td>The existing Stamford Primary School nearby ha...</td>
      <td>19</td>
      <td>hi7lm8f</td>
      <td>qgo2dz</td>
      <td>jmzyn</td>
      <td>24</td>
      <td>1.635310e+09</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20</td>
      <td>---\n1.0.2 | [Source code](https://github.com/...</td>
      <td>20</td>
      <td>hi7gzzk</td>
      <td>qgo2dz</td>
      <td>sneakpeek_bot</td>
      <td>5</td>
      <td>1.635307e+09</td>
    </tr>
    <tr>
      <th>2</th>
      <td>219</td>
      <td>Yup - or they can decide not to get married [l...</td>
      <td>219</td>
      <td>hi83v1u</td>
      <td>hi7pfvf</td>
      <td>sitsthewind</td>
      <td>2</td>
      <td>1.635324e+09</td>
    </tr>
    <tr>
      <th>3</th>
      <td>322</td>
      <td>https://www.ons.gov.uk/peoplepopulationandcomm...</td>
      <td>322</td>
      <td>hi92uhw</td>
      <td>hi80uyy</td>
      <td>xvdrk</td>
      <td>1</td>
      <td>1.635345e+09</td>
    </tr>
    <tr>
      <th>4</th>
      <td>325</td>
      <td>https://www.iras.gov.sg/taxes/individual-incom...</td>
      <td>325</td>
      <td>hi7z0ep</td>
      <td>hi7xn9n</td>
      <td>GoodBoysThinkAlike</td>
      <td>30</td>
      <td>1.635320e+09</td>
    </tr>
    <tr>
      <th>5</th>
      <td>329</td>
      <td>I will be messaging you in 11 years on [**2032...</td>
      <td>329</td>
      <td>hi7qeza</td>
      <td>hi7qcsv</td>
      <td>RemindMeBot</td>
      <td>2</td>
      <td>1.635313e+09</td>
    </tr>
    <tr>
      <th>6</th>
      <td>329</td>
      <td>[**CLICK THIS LINK**](https://www.reddit.com/m...</td>
      <td>329</td>
      <td>hi7qeza</td>
      <td>hi7qcsv</td>
      <td>RemindMeBot</td>
      <td>2</td>
      <td>1.635313e+09</td>
    </tr>
    <tr>
      <th>7</th>
      <td>329</td>
      <td>^(Parent commenter can ) [^(delete this messag...</td>
      <td>329</td>
      <td>hi7qeza</td>
      <td>hi7qcsv</td>
      <td>RemindMeBot</td>
      <td>2</td>
      <td>1.635313e+09</td>
    </tr>
    <tr>
      <th>8</th>
      <td>329</td>
      <td>|[^(Info)](https://www.reddit.com/r/RemindMeBo...</td>
      <td>329</td>
      <td>hi7qeza</td>
      <td>hi7qcsv</td>
      <td>RemindMeBot</td>
      <td>2</td>
      <td>1.635313e+09</td>
    </tr>
    <tr>
      <th>9</th>
      <td>335</td>
      <td>https://www.reddit.com/r/singapore/comments/8n...</td>
      <td>335</td>
      <td>hi7y3w6</td>
      <td>hi7xoes</td>
      <td>yewjrn</td>
      <td>5</td>
      <td>1.635319e+09</td>
    </tr>
    <tr>
      <th>10</th>
      <td>350</td>
      <td>you [said](https://old.reddit.com/r/singapore/...</td>
      <td>350</td>
      <td>hi9hc8x</td>
      <td>hi9bah8</td>
      <td>ILikeWhiteMen</td>
      <td>3</td>
      <td>1.635351e+09</td>
    </tr>
  </tbody>
</table>
</div>




```python
# define function to remove both links and markup links 
# also remove \' from dont\'t
def remove_https(item):
    #remove https links
    item_1 = re.sub(r"[(+*)]\S*https?:\S*[(+*)]", "", item)
    #remove https links with no brackets
    item_2 = re.sub('http://\S+|https://\S+', " ", item_1)
    #remove link markups []
    #note that this will also remove comment fields with ["Delete"] 
    item_3 = re.sub(r"[\(\[].*?[\)\]]", " ", item_2)
#     #remove \ in don\'t
#     item_4 = re.sub("[\"\']", "'", item_3)
    return item_3

df_comment["comment"] = [remove_https(x) for x in df_comment.comment]
```


```python
#check the temporary table to see if links/ markuplinks/ \' 
#all links has been removed
#unecessary comments (highlighted in yellow) can be removed later by filtering out unecessary usernames
df_temp["comment"] = [remove_https(x) for x in df_temp.comment]
df_temp.style.apply(lambda x: ['background: lightyellow' if x.username == "RemindMeBot" \
                               or x.username =="sneakpeek_bot" else '' for i in x], axis=1)
```




<style type="text/css">
#T_1c78c_row1_col0, #T_1c78c_row1_col1, #T_1c78c_row1_col2, #T_1c78c_row1_col3, #T_1c78c_row1_col4, #T_1c78c_row1_col5, #T_1c78c_row1_col6, #T_1c78c_row1_col7, #T_1c78c_row5_col0, #T_1c78c_row5_col1, #T_1c78c_row5_col2, #T_1c78c_row5_col3, #T_1c78c_row5_col4, #T_1c78c_row5_col5, #T_1c78c_row5_col6, #T_1c78c_row5_col7, #T_1c78c_row6_col0, #T_1c78c_row6_col1, #T_1c78c_row6_col2, #T_1c78c_row6_col3, #T_1c78c_row6_col4, #T_1c78c_row6_col5, #T_1c78c_row6_col6, #T_1c78c_row6_col7, #T_1c78c_row7_col0, #T_1c78c_row7_col1, #T_1c78c_row7_col2, #T_1c78c_row7_col3, #T_1c78c_row7_col4, #T_1c78c_row7_col5, #T_1c78c_row7_col6, #T_1c78c_row7_col7, #T_1c78c_row8_col0, #T_1c78c_row8_col1, #T_1c78c_row8_col2, #T_1c78c_row8_col3, #T_1c78c_row8_col4, #T_1c78c_row8_col5, #T_1c78c_row8_col6, #T_1c78c_row8_col7 {
  background: lightyellow;
}
</style>
<table id="T_1c78c">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_1c78c_level0_col0" class="col_heading level0 col0" >index_name</th>
      <th id="T_1c78c_level0_col1" class="col_heading level0 col1" >comment</th>
      <th id="T_1c78c_level0_col2" class="col_heading level0 col2" >index</th>
      <th id="T_1c78c_level0_col3" class="col_heading level0 col3" >comment_id</th>
      <th id="T_1c78c_level0_col4" class="col_heading level0 col4" >parent_id</th>
      <th id="T_1c78c_level0_col5" class="col_heading level0 col5" >username</th>
      <th id="T_1c78c_level0_col6" class="col_heading level0 col6" >upvotes</th>
      <th id="T_1c78c_level0_col7" class="col_heading level0 col7" >datetime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_1c78c_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_1c78c_row0_col0" class="data row0 col0" >19</td>
      <td id="T_1c78c_row0_col1" class="data row0 col1" >The existing Stamford Primary School nearby has already been slated to merge to Farrer Park Primary School campus in  . Someone obviously didn't do homework!</td>
      <td id="T_1c78c_row0_col2" class="data row0 col2" >19</td>
      <td id="T_1c78c_row0_col3" class="data row0 col3" >hi7lm8f</td>
      <td id="T_1c78c_row0_col4" class="data row0 col4" >qgo2dz</td>
      <td id="T_1c78c_row0_col5" class="data row0 col5" >jmzyn</td>
      <td id="T_1c78c_row0_col6" class="data row0 col6" >24</td>
      <td id="T_1c78c_row0_col7" class="data row0 col7" >1635310102.000000</td>
    </tr>
    <tr>
      <th id="T_1c78c_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_1c78c_row1_col0" class="data row1 col0" >20</td>
      <td id="T_1c78c_row1_col1" class="data row1 col1" >---
1.0.2 |   |  </td>
      <td id="T_1c78c_row1_col2" class="data row1 col2" >20</td>
      <td id="T_1c78c_row1_col3" class="data row1 col3" >hi7gzzk</td>
      <td id="T_1c78c_row1_col4" class="data row1 col4" >qgo2dz</td>
      <td id="T_1c78c_row1_col5" class="data row1 col5" >sneakpeek_bot</td>
      <td id="T_1c78c_row1_col6" class="data row1 col6" >5</td>
      <td id="T_1c78c_row1_col7" class="data row1 col7" >1635307441.000000</td>
    </tr>
    <tr>
      <th id="T_1c78c_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_1c78c_row2_col0" class="data row2 col0" >219</td>
      <td id="T_1c78c_row2_col1" class="data row2 col1" >Yup - or they can decide not to get married  .</td>
      <td id="T_1c78c_row2_col2" class="data row2 col2" >219</td>
      <td id="T_1c78c_row2_col3" class="data row2 col3" >hi83v1u</td>
      <td id="T_1c78c_row2_col4" class="data row2 col4" >hi7pfvf</td>
      <td id="T_1c78c_row2_col5" class="data row2 col5" >sitsthewind</td>
      <td id="T_1c78c_row2_col6" class="data row2 col6" >2</td>
      <td id="T_1c78c_row2_col7" class="data row2 col7" >1635324215.000000</td>
    </tr>
    <tr>
      <th id="T_1c78c_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_1c78c_row3_col0" class="data row3 col0" >322</td>
      <td id="T_1c78c_row3_col1" class="data row3 col1" > </td>
      <td id="T_1c78c_row3_col2" class="data row3 col2" >322</td>
      <td id="T_1c78c_row3_col3" class="data row3 col3" >hi92uhw</td>
      <td id="T_1c78c_row3_col4" class="data row3 col4" >hi80uyy</td>
      <td id="T_1c78c_row3_col5" class="data row3 col5" >xvdrk</td>
      <td id="T_1c78c_row3_col6" class="data row3 col6" >1</td>
      <td id="T_1c78c_row3_col7" class="data row3 col7" >1635345487.000000</td>
    </tr>
    <tr>
      <th id="T_1c78c_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_1c78c_row4_col0" class="data row4 col0" >325</td>
      <td id="T_1c78c_row4_col1" class="data row4 col1" > </td>
      <td id="T_1c78c_row4_col2" class="data row4 col2" >325</td>
      <td id="T_1c78c_row4_col3" class="data row4 col3" >hi7z0ep</td>
      <td id="T_1c78c_row4_col4" class="data row4 col4" >hi7xn9n</td>
      <td id="T_1c78c_row4_col5" class="data row4 col5" >GoodBoysThinkAlike</td>
      <td id="T_1c78c_row4_col6" class="data row4 col6" >30</td>
      <td id="T_1c78c_row4_col7" class="data row4 col7" >1635320022.000000</td>
    </tr>
    <tr>
      <th id="T_1c78c_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_1c78c_row5_col0" class="data row5 col0" >329</td>
      <td id="T_1c78c_row5_col1" class="data row5 col1" >I will be messaging you in 11 years on [**2032-10-27 05:40:25 UTC to remind you of [**this link</td>
      <td id="T_1c78c_row5_col2" class="data row5 col2" >329</td>
      <td id="T_1c78c_row5_col3" class="data row5 col3" >hi7qeza</td>
      <td id="T_1c78c_row5_col4" class="data row5 col4" >hi7qcsv</td>
      <td id="T_1c78c_row5_col5" class="data row5 col5" >RemindMeBot</td>
      <td id="T_1c78c_row5_col6" class="data row5 col6" >2</td>
      <td id="T_1c78c_row5_col7" class="data row5 col7" >1635313267.000000</td>
    </tr>
    <tr>
      <th id="T_1c78c_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_1c78c_row6_col0" class="data row6 col0" >329</td>
      <td id="T_1c78c_row6_col1" class="data row6 col1" >[**CLICK THIS LINK to send a PM to also be reminded and to reduce spam.</td>
      <td id="T_1c78c_row6_col2" class="data row6 col2" >329</td>
      <td id="T_1c78c_row6_col3" class="data row6 col3" >hi7qeza</td>
      <td id="T_1c78c_row6_col4" class="data row6 col4" >hi7qcsv</td>
      <td id="T_1c78c_row6_col5" class="data row6 col5" >RemindMeBot</td>
      <td id="T_1c78c_row6_col6" class="data row6 col6" >2</td>
      <td id="T_1c78c_row6_col7" class="data row6 col7" >1635313267.000000</td>
    </tr>
    <tr>
      <th id="T_1c78c_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_1c78c_row7_col0" class="data row7 col0" >329</td>
      <td id="T_1c78c_row7_col1" class="data row7 col1" >^  [^(delete this message to hide from others.</td>
      <td id="T_1c78c_row7_col2" class="data row7 col2" >329</td>
      <td id="T_1c78c_row7_col3" class="data row7 col3" >hi7qeza</td>
      <td id="T_1c78c_row7_col4" class="data row7 col4" >hi7qcsv</td>
      <td id="T_1c78c_row7_col5" class="data row7 col5" >RemindMeBot</td>
      <td id="T_1c78c_row7_col6" class="data row7 col6" >2</td>
      <td id="T_1c78c_row7_col7" class="data row7 col7" >1635313267.000000</td>
    </tr>
    <tr>
      <th id="T_1c78c_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_1c78c_row8_col0" class="data row8 col0" >329</td>
      <td id="T_1c78c_row8_col1" class="data row8 col1" >|[^Your Reminders|
|-|-|-|-|</td>
      <td id="T_1c78c_row8_col2" class="data row8 col2" >329</td>
      <td id="T_1c78c_row8_col3" class="data row8 col3" >hi7qeza</td>
      <td id="T_1c78c_row8_col4" class="data row8 col4" >hi7qcsv</td>
      <td id="T_1c78c_row8_col5" class="data row8 col5" >RemindMeBot</td>
      <td id="T_1c78c_row8_col6" class="data row8 col6" >2</td>
      <td id="T_1c78c_row8_col7" class="data row8 col7" >1635313267.000000</td>
    </tr>
    <tr>
      <th id="T_1c78c_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_1c78c_row9_col0" class="data row9 col0" >335</td>
      <td id="T_1c78c_row9_col1" class="data row9 col1" > </td>
      <td id="T_1c78c_row9_col2" class="data row9 col2" >335</td>
      <td id="T_1c78c_row9_col3" class="data row9 col3" >hi7y3w6</td>
      <td id="T_1c78c_row9_col4" class="data row9 col4" >hi7xoes</td>
      <td id="T_1c78c_row9_col5" class="data row9 col5" >yewjrn</td>
      <td id="T_1c78c_row9_col6" class="data row9 col6" >5</td>
      <td id="T_1c78c_row9_col7" class="data row9 col7" >1635319252.000000</td>
    </tr>
    <tr>
      <th id="T_1c78c_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_1c78c_row10_col0" class="data row10 col0" >350</td>
      <td id="T_1c78c_row10_col1" class="data row10 col1" >you   "If you own your own company, you could just take a loan   as shareholder at 0% interest rate to cover the payment. Alternatively, just take a personal loan collateralized by your assets   from a bank. Just need to ensure it doesn't violate the debt limits re: HDB but still there should be ways to get around the salary ceiling. That's precisely why most American billionaires don't pay tax, they just borrow against their shares."</td>
      <td id="T_1c78c_row10_col2" class="data row10 col2" >350</td>
      <td id="T_1c78c_row10_col3" class="data row10 col3" >hi9hc8x</td>
      <td id="T_1c78c_row10_col4" class="data row10 col4" >hi9bah8</td>
      <td id="T_1c78c_row10_col5" class="data row10 col5" >ILikeWhiteMen</td>
      <td id="T_1c78c_row10_col6" class="data row10 col6" >3</td>
      <td id="T_1c78c_row10_col7" class="data row10 col7" >1635351397.000000</td>
    </tr>
  </tbody>
</table>





```python
#check number of rows
len(df_comment)
```




    580




```python
#implode and remove column index_name
df_comment_1 = df_comment.groupby("index_name")["comment"].apply(lambda x: " ".join(x)).reset_index().drop("index_name", axis = 1)
```


```python
#check that total columns are still the same before explode
#seems that there is additional row 
len(df_comment_1)
```




    359




```python
#last row is the additional empty row
df_comment_1.tail()
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
      <th>comment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>354</th>
      <td>What do you think?</td>
    </tr>
    <tr>
      <th>355</th>
      <td>By your logic rich people shouldn't pay taxes ...</td>
    </tr>
    <tr>
      <th>356</th>
      <td>Well, their future babies tax dollars are gonn...</td>
    </tr>
    <tr>
      <th>357</th>
      <td>No. Statistics matter when drawing a conclusio...</td>
    </tr>
    <tr>
      <th>358</th>
      <td>/Users/puttasathvik/Downloads/TextBasedAnalysi...</td>
    </tr>
  </tbody>
</table>
</div>




```python
#remove the last row by slicing 
df_comment_1 = df_comment_1[:358]
```

## 3b. Removing Deleted/Bot User Accounts and Empty Comments

After tidying up df_comment, we can combine it with df_details to further clean the data. This process will involve removing unwanted usernames (like bot accounts) and any empty comments. During the cleanup of df_comment, we made sure the number of rows stayed the same. This allows us to safely merge the two datasets because their indexes will match perfectly.


```python
#concatenate both datasets with same index 
df = pd.concat([df_comment_1, df_details], axis = 1)
df
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
      <th>comment</th>
      <th>comment_id</th>
      <th>parent_id</th>
      <th>username</th>
      <th>upvotes</th>
      <th>datetime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Honestly pretty happy with this - getting a pr...</td>
      <td>hi7j3kg</td>
      <td>qgo2dz</td>
      <td>eastsidegoondu</td>
      <td>192</td>
      <td>1.635309e+09</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Key part here is that it's also unavailable fo...</td>
      <td>hi7hcbh</td>
      <td>qgo2dz</td>
      <td>MacWithoutCheese</td>
      <td>505</td>
      <td>1.635308e+09</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Great. We should stop treating public housing ...</td>
      <td>hi7i2xs</td>
      <td>qgo2dz</td>
      <td>pewsg</td>
      <td>314</td>
      <td>1.635308e+09</td>
    </tr>
    <tr>
      <th>3</th>
      <td>This is good news for those who really needs a...</td>
      <td>hi7ieqf</td>
      <td>qgo2dz</td>
      <td>PossibleConsistent77</td>
      <td>97</td>
      <td>1.635308e+09</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Never see Noobhdbbuyer commenting if he's happ...</td>
      <td>hi7mdia</td>
      <td>qgo2dz</td>
      <td>Exofanjongdae</td>
      <td>60</td>
      <td>1.635311e+09</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>353</th>
      <td>I dare you to write to Straits Times forum pro...</td>
      <td>hi84isq</td>
      <td>hi81api</td>
      <td>sitsthewind</td>
      <td>4</td>
      <td>1.635325e+09</td>
    </tr>
    <tr>
      <th>354</th>
      <td>What do you think?</td>
      <td>hi81lym</td>
      <td>hi81api</td>
      <td>loveforlandlords</td>
      <td>-1</td>
      <td>1.635322e+09</td>
    </tr>
    <tr>
      <th>355</th>
      <td>By your logic rich people shouldn't pay taxes ...</td>
      <td>hi88k5u</td>
      <td>hi88fn0</td>
      <td>loveforlandlords</td>
      <td>4</td>
      <td>1.635328e+09</td>
    </tr>
    <tr>
      <th>356</th>
      <td>Well, their future babies tax dollars are gonn...</td>
      <td>hi8mfr1</td>
      <td>hi88fn0</td>
      <td>ikanjonnies</td>
      <td>3</td>
      <td>1.635338e+09</td>
    </tr>
    <tr>
      <th>357</th>
      <td>No. Statistics matter when drawing a conclusio...</td>
      <td>hidcurl</td>
      <td>hib4b48</td>
      <td>PriceToBookValue</td>
      <td>1</td>
      <td>1.635423e+09</td>
    </tr>
  </tbody>
</table>
<p>358 rows × 6 columns</p>
</div>




```python
#before we have replaced comments with empty string 
#see how many rows there are 
len(df[df.comment == " "])
```




    10




```python
df[df.comment == " "]
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
      <th>comment</th>
      <th>comment_id</th>
      <th>parent_id</th>
      <th>username</th>
      <th>upvotes</th>
      <th>datetime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>52</th>
      <td></td>
      <td>hi7pcje</td>
      <td>qgo2dz</td>
      <td>NaN</td>
      <td>-6</td>
      <td>1.635313e+09</td>
    </tr>
    <tr>
      <th>78</th>
      <td></td>
      <td>hi7q2m5</td>
      <td>hi7i9ej</td>
      <td>NaN</td>
      <td>17</td>
      <td>1.635313e+09</td>
    </tr>
    <tr>
      <th>156</th>
      <td></td>
      <td>hi7xw5v</td>
      <td>hi7tdh4</td>
      <td>NaN</td>
      <td>10</td>
      <td>1.635319e+09</td>
    </tr>
    <tr>
      <th>186</th>
      <td></td>
      <td>hi7khkd</td>
      <td>hi7k5ie</td>
      <td>NaN</td>
      <td>32</td>
      <td>1.635309e+09</td>
    </tr>
    <tr>
      <th>198</th>
      <td></td>
      <td>hi7oufu</td>
      <td>hi7mxzs</td>
      <td>NaN</td>
      <td>1</td>
      <td>1.635312e+09</td>
    </tr>
    <tr>
      <th>228</th>
      <td></td>
      <td>hi7lomx</td>
      <td>hi7lgl9</td>
      <td>NaN</td>
      <td>7</td>
      <td>1.635310e+09</td>
    </tr>
    <tr>
      <th>264</th>
      <td></td>
      <td>hi7z5qj</td>
      <td>hi7ys7b</td>
      <td>NaN</td>
      <td>5</td>
      <td>1.635320e+09</td>
    </tr>
    <tr>
      <th>282</th>
      <td></td>
      <td>hi89j0o</td>
      <td>hi8953v</td>
      <td>NaN</td>
      <td>9</td>
      <td>1.635329e+09</td>
    </tr>
    <tr>
      <th>293</th>
      <td></td>
      <td>hi7ptfn</td>
      <td>hi7p79j</td>
      <td>NaN</td>
      <td>1</td>
      <td>1.635313e+09</td>
    </tr>
    <tr>
      <th>335</th>
      <td></td>
      <td>hi7y3w6</td>
      <td>hi7xoes</td>
      <td>yewjrn</td>
      <td>5</td>
      <td>1.635319e+09</td>
    </tr>
  </tbody>
</table>
</div>




```python
#remove rows with empty field under comment attribute
#removed 10 rows, 348 rows
df = df[df.comment != " "]
```


```python
#username is none == useraccount that has been deleted
#seems like deleted accounts has comments that is not removed
#we will leave these accounts alone 
df[df.username.isna()]
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
      <th>comment</th>
      <th>comment_id</th>
      <th>parent_id</th>
      <th>username</th>
      <th>upvotes</th>
      <th>datetime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>21</th>
      <td>Hear me out. What if they gave out higher subs...</td>
      <td>hi81l0r</td>
      <td>qgo2dz</td>
      <td>NaN</td>
      <td>4</td>
      <td>1.635322e+09</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Oh no. how am I going to be a millionaire now?</td>
      <td>hi7otkz</td>
      <td>qgo2dz</td>
      <td>NaN</td>
      <td>1</td>
      <td>1.635312e+09</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[(df.username == "RemindMeBot") | (df.username == "sneakpeek_bot")]
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
      <th>comment</th>
      <th>comment_id</th>
      <th>parent_id</th>
      <th>username</th>
      <th>upvotes</th>
      <th>datetime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20</th>
      <td>---\n1.0.2 |   |</td>
      <td>hi7gzzk</td>
      <td>qgo2dz</td>
      <td>sneakpeek_bot</td>
      <td>5</td>
      <td>1.635307e+09</td>
    </tr>
    <tr>
      <th>329</th>
      <td>I will be messaging you in 11 years on [**2032...</td>
      <td>hi7qeza</td>
      <td>hi7qcsv</td>
      <td>RemindMeBot</td>
      <td>2</td>
      <td>1.635313e+09</td>
    </tr>
  </tbody>
</table>
</div>




```python
#remove comments by bots "RemindMeBot", "sneakpeek_bot"
#reset index so that it is running 
#removeed 2 rows, resulting dataframe will have 346 rows
df = df[(df.username != "RemindMeBot") & (df.username != "sneakpeek_bot")].reset_index(drop = True)
```


```python
df.head()
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
      <th>comment</th>
      <th>comment_id</th>
      <th>parent_id</th>
      <th>username</th>
      <th>upvotes</th>
      <th>datetime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Honestly pretty happy with this - getting a pr...</td>
      <td>hi7j3kg</td>
      <td>qgo2dz</td>
      <td>eastsidegoondu</td>
      <td>192</td>
      <td>1.635309e+09</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Key part here is that it's also unavailable fo...</td>
      <td>hi7hcbh</td>
      <td>qgo2dz</td>
      <td>MacWithoutCheese</td>
      <td>505</td>
      <td>1.635308e+09</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Great. We should stop treating public housing ...</td>
      <td>hi7i2xs</td>
      <td>qgo2dz</td>
      <td>pewsg</td>
      <td>314</td>
      <td>1.635308e+09</td>
    </tr>
    <tr>
      <th>3</th>
      <td>This is good news for those who really needs a...</td>
      <td>hi7ieqf</td>
      <td>qgo2dz</td>
      <td>PossibleConsistent77</td>
      <td>97</td>
      <td>1.635308e+09</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Never see Noobhdbbuyer commenting if he's happ...</td>
      <td>hi7mdia</td>
      <td>qgo2dz</td>
      <td>Exofanjongdae</td>
      <td>60</td>
      <td>1.635311e+09</td>
    </tr>
  </tbody>
</table>
</div>




```python
len(df)
```




    346



## 3c. Text Clean-Up

Text clean-up is part of getting data ready. Online comments are often messy and have issues like:

1. Typos

2. Contractions (don’t, can’t, shouldn’t)

3. Abbreviations (govt for government, info for information)

4. Internet slang (irl, fyi)

5. Creole languages (like Singlish in Singapore)

We need to fix these problems to turn the text into proper English without slang or grammar mistakes. Also, since we will use TextBlob for analyzing feelings in the text, we need to change emojis into words or phrases because TextBlob only works with text.









```python
#ensure that comment attribute is of correct data type
df["comment"] = df.comment.astype("str")
df["comment"] = [item.lower() for item in df.comment]

#remove apostrophe at the beginning and end of each word (e.g. 'like, 'this, or', this')
df["comment"] = [re.sub(r"(\B'\b)|(\b'\B)", ' ', item) for item in df.comment]
df["comment"] = [re.sub(r'…', ' ', item) for item in df.comment]
df["comment"] = [item.replace('\\',' ') for item in df.comment]
df["comment"] = [item.replace('/',' ') for item in df.comment]
```


```python
#TEST
#have a overview/ general sensing of types of contractions we have 
#create a temp list of tokenized sentences 
df_token_temp = [item.split() for item in df["comment"]]
df_token_temp = [i for word in df_token_temp for i in word]
df_contraction_temp = [re.findall("(?=\S*['-])([a-zA-Z'-]+)", i) for i in df_token_temp]
df_contraction_temp_1 = [i for item in df_contraction_temp if item != [] for i in item]
df_contraction_temp_2 = [i for n, i in enumerate(df_contraction_temp_1) if i not in df_contraction_temp_1[:n]]
print(df_contraction_temp_2)
```

    ['-', "don't", "it's", "he's", 'no-mop', '-year', "i'm", 'rent-free', "didn't", "doesn't", "hasn't", "we'll", 'instrument', '--especially', '-it', 'generation', '--like', "isn't", 'confidence', '-one', "that's", 'non-nuclear', "they'll", "what's", "won't", "they're", 'non-married', "people's", "can't", "there'll", 'split-society-haves-vs-have-nots', 'hdb-dwellers', "wouldn't", "plh's", 'non-prime', 'win-win', 'so-called', "other's", "they'd", "shouldn't", 'senior-friendly', "let's", 'knock-on', "you're", "it'll", "friend's", "wife's", 'prime-ish', 'in-between', '-room', 'non-mature', '-rooms', "there's", 'family-building', "worker's", "i'll", 'non-matured', 'pap-land', 'lgbtq-friendly', 'single-friendly', 'sub-', 'dual-income', 'en-bloc', "we're", 'family-only', "couple's", "government's", 'self-employed', "haven't", 'non-flippers', "i've", "kpkb'ing", "aren't", "sg's", 'pre-marriage', "you'll", "govenment's", "she's", 'non-high', 'non-rental', 'higher-paid', 'lower-paid', 'covid-']
    


```python
#define a function to clean up these contractions 
#for \'s such as he's, she's we will just replace with he and she as is is a stop word and will be removed 
def decontract(phrase):  
    phrase = re.sub(r"can\'t", "cannot", phrase)
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"let\'s", "let us", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'s", "", phrase)
    #"kpkb'ing"
    phrase = re.sub(r"\'ing", "", phrase)
    
    phrase = re.sub(r"can’t", "cannot", phrase)
    phrase = re.sub(r"won’t", "will not", phrase)
    phrase = re.sub(r"let’s", "let us", phrase)
    phrase = re.sub(r"n’t", " not", phrase)
    phrase = re.sub(r"’m", " am", phrase)
    phrase = re.sub(r"’ll", " will", phrase)
    phrase = re.sub(r"’re", " are", phrase)
    phrase = re.sub(r"’d", " would", phrase)
    phrase = re.sub(r"’ve", " have", phrase)
    phrase = re.sub(r"’s", "", phrase)
    #"kpkb'ing"
    phrase = re.sub(r"’ing", "", phrase)
    return phrase
```


```python
#TEST
#test it on df_contraction_temp_2
df_contraction_temp_3 = [decontract(i) for i in df_contraction_temp_2]
print(df_contraction_temp_3)
```

    ['-', 'do not', 'it', 'he', 'no-mop', '-year', 'i am', 'rent-free', 'did not', 'does not', 'has not', 'we will', 'instrument', '--especially', '-it', 'generation', '--like', 'is not', 'confidence', '-one', 'that', 'non-nuclear', 'they will', 'what', 'will not', 'they are', 'non-married', 'people', 'cannot', 'there will', 'split-society-haves-vs-have-nots', 'hdb-dwellers', 'would not', 'plh', 'non-prime', 'win-win', 'so-called', 'other', 'they would', 'should not', 'senior-friendly', 'let us', 'knock-on', 'you are', 'it will', 'friend', 'wife', 'prime-ish', 'in-between', '-room', 'non-mature', '-rooms', 'there', 'family-building', 'worker', 'i will', 'non-matured', 'pap-land', 'lgbtq-friendly', 'single-friendly', 'sub-', 'dual-income', 'en-bloc', 'we are', 'family-only', 'couple', 'government', 'self-employed', 'have not', 'non-flippers', 'i have', 'kpkb', 'are not', 'sg', 'pre-marriage', 'you will', 'govenment', 'she', 'non-high', 'non-rental', 'higher-paid', 'lower-paid', 'covid-']
    


```python
#decontract words in dataframe
df["comment"] = [decontract(i) for i in df.comment]
```


```python
#define a function to find all emojis
def extract_emojis(s):
    return ''.join(c for c in s if c in emoji.UNICODE_EMOJI['en'])
```


```python
import emoji

def extract_emojis(s):
    return ''.join(c for c in s if c in emoji.EMOJI_DATA)
# Assuming df is your DataFrame and 'comment' is the column with the text
emoji_lst = [extract_emojis(i) for i in df['comment'].tolist()]
emoji_lst = list(filter(None, emoji_lst))
emoji_lst
```




    ['😢', '🤨', '🙄', '😂']




```python
#define a function that converts emojis to words/ phrase
def convert_emoji(phrase):
    phrase = re.sub(r"😢", " sad ", phrase)
    phrase = re.sub(r"🤨", " not confident ", phrase)
    phrase = re.sub(r"🙄", " annoying ",  phrase)
    phrase = re.sub(r"😂", " laugh ", phrase)
    return phrase

df["comment"] = [convert_emoji(i) for i in df.comment]

```


```python
#define a function that converts all typos
def clean_typo(phrase):
    phrase = re.sub(r"-ish", "", phrase)
    phrase = re.sub(r"rrcent", "recent", phrase)
    phrase = re.sub(r"govenment", " government ", phrase)
    phrase = re.sub(r"diffit", "definitely", phrase)
    phrase = re.sub(r"overexxagearting", "overexaggerate", phrase)
    phrase = re.sub(r"en-bloc", "enbloc", phrase)
    phrase = re.sub(r" dnt ", "do not", phrase)
    phrase = re.sub(r" underdeclared ", " under declare ", phrase)
    phrase = re.sub(r" lgbt ", " lgbtq ", phrase)
    phrase = re.sub(r" 9wn ", " own ", phrase)
    phrase = re.sub(r" rocher ", " rochor ", phrase)
    phrase = re.sub(r" pinnacle ", " duxton ", phrase)
    phrase = re.sub(r" cdb ", " cbd ", phrase)
    phrase = re.sub(r" hivemind ", " hive mind ", phrase)
    phrase = re.sub(r" claw back ", " clawback ", phrase)
    phrase = re.sub(r" discludes ", " excludes ", phrase)
    phrase = re.sub(r" hugeee ", " huge ", phrase)
    phrase = re.sub(r" birthrate ", " birth rate ", phrase)
    phrase = re.sub(r" oligations ", " obligations ", phrase)
    phrase = re.sub(r" wayyy ", " way ", phrase)
    phrase = re.sub(r" plhs ", " plh ", phrase)
    phrase = re.sub(r" noobhdbbuyer ", " noob hdb buyer ", phrase)
    return phrase
    
df["comment"] = [clean_typo(i) for i in df.comment]
```


```python
#define a function to convert all short-forms/ short terms 
def clean_short(phrase):
    phrase = re.sub(r"fyi", "for your information", phrase)
    phrase = re.sub(r"tbh", "to be honest", phrase)
    phrase = re.sub(r" esp ", " especially ", phrase)
    phrase = re.sub(r" info ", "information", phrase)
    phrase = re.sub(r"gonna", "going to", phrase)
    phrase = re.sub(r"stats", "statistics", phrase)
    phrase = re.sub(r"rm ", " room ", phrase)
    phrase = phrase.replace("i.e.", " ")
    phrase = re.sub(r"idk", "i do not know", phrase)
    phrase = re.sub(r"haha", "laugh", phrase)
    phrase = re.sub(r"yr", " year", phrase)
    phrase = re.sub(r" sg ", " singapore ", phrase)
    phrase = re.sub(r" mil ", " million ", phrase)
    phrase = re.sub(r" =", " same ", phrase)
    phrase = re.sub(r" msr. ", " mortage serving ratio ", phrase)
    phrase = re.sub(r" eip ", " ethnic integration policy ", phrase)
    phrase = re.sub(r" g ", " government ", phrase)
    phrase = re.sub(r"^imo ", " in my opinion ", phrase)
    phrase = re.sub(r" pp ", " private property ", phrase)
    phrase = re.sub(r" grad ", " graduate ", phrase)
    phrase = re.sub(r" ns ", " national service ", phrase)
    phrase = re.sub(r" bc ", " because ", phrase)
    phrase = re.sub(r" u ", " you ", phrase)
    phrase = re.sub(r" ur ", " your ", phrase)
    phrase = re.sub(r"^yo ", " year ", phrase)
    phrase = re.sub(r" vs ", " versus ", phrase)
    phrase = re.sub(r" irl ", " in reality ", phrase)
    phrase = re.sub(r" tfr ", " total fertility rate ", phrase)
    phrase = re.sub(r" fk ", " fuck ", phrase)
    phrase = re.sub(r" fked ", " fuck ", phrase)
    phrase = re.sub(r" fucked ", " fuck ", phrase)
    phrase = re.sub(r".  um.", " cynical. ", phrase)
    phrase = re.sub(r" pre-", " before ", phrase)
    phrase = re.sub(r" ed ", " education ", phrase)
    return phrase
```


```python
#define a function that converts singlish
def singlish_clean(phrase):
    phrase = re.sub(r"yup", " yes", phrase)
    phrase = re.sub(r" yah ", " yes", phrase)
    phrase = re.sub(r"yeah", "yes", phrase)
    phrase = re.sub(r" ya ", "  yes", phrase)
    phrase = re.sub(r"song ah", "good", phrase)
    phrase = re.sub(r" lah", " ", phrase)
    phrase = re.sub(r"hurray", "congratulation", phrase)
    phrase = re.sub(r"^um", "unsure", phrase)
    phrase = re.sub(r" sian ", " tired of ", phrase)
    phrase = re.sub(r" eh", " ", phrase)
    phrase = re.sub(r" hentak kaki ", " stagnant ", phrase)
    phrase = re.sub(r" ulu ", " remote ", phrase)
    phrase = re.sub(r" kpkb ", " complain ", phrase)
    phrase = re.sub(r" leh.", " .", phrase)
    phrase = re.sub(r"sinkies", " rude ", phrase)
    phrase = re.sub(r"sinkie", " rude ", phrase)
    phrase = re.sub(r"shitty", "shit", phrase)
    return phrase

df["comment"] = [singlish_clean(i) for i in df.comment]
```


```python
def others_clean(phrase):
    phrase = re.sub(r" govt ", " government ", phrase)
    phrase = re.sub(r"14 000", "14k", phrase)
    phrase = re.sub(r"14000", "14k", phrase)
    phrase = re.sub(r"14,000", "14k", phrase)
    phrase = re.sub(r"flipper", "flip ", phrase)
    phrase = re.sub(r"flip s", "flip", phrase)
    phrase = re.sub(r"flipping", "flip ", phrase)
    phrase = re.sub(r"hdbs", "hdb", phrase)
    phrase = re.sub(r"dont", "do not", phrase)
    phrase = re.sub(r"cant", "cannot", phrase)
    phrase = re.sub(r"shouldnt", "should not", phrase)
    phrase = re.sub(r"condominiums", "condo ", phrase)
    phrase = re.sub(r"condominium", "condo ", phrase)
    phrase = re.sub(r"btos", "bto", phrase)
    phrase = re.sub(r"non-", "not ", phrase)
    phrase = re.sub(r" x+ ", " ", phrase)
    phrase = re.sub(r" ccr or ", " ", phrase)
    phrase = re.sub(r" its ", " it ", phrase)
    return phrase

df["comment"] = [others_clean(i) for i in df.comment]
```


```python
df.head()
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
      <th>comment</th>
      <th>comment_id</th>
      <th>parent_id</th>
      <th>username</th>
      <th>upvotes</th>
      <th>datetime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>honestly pretty happy with this - getting a pr...</td>
      <td>hi7j3kg</td>
      <td>qgo2dz</td>
      <td>eastsidegoondu</td>
      <td>192</td>
      <td>1.635309e+09</td>
    </tr>
    <tr>
      <th>1</th>
      <td>key part here is that it also unavailable for ...</td>
      <td>hi7hcbh</td>
      <td>qgo2dz</td>
      <td>MacWithoutCheese</td>
      <td>505</td>
      <td>1.635308e+09</td>
    </tr>
    <tr>
      <th>2</th>
      <td>great. we should stop treating public housing ...</td>
      <td>hi7i2xs</td>
      <td>qgo2dz</td>
      <td>pewsg</td>
      <td>314</td>
      <td>1.635308e+09</td>
    </tr>
    <tr>
      <th>3</th>
      <td>this is good news for those who really needs a...</td>
      <td>hi7ieqf</td>
      <td>qgo2dz</td>
      <td>PossibleConsistent77</td>
      <td>97</td>
      <td>1.635308e+09</td>
    </tr>
    <tr>
      <th>4</th>
      <td>never see noob hdb buyer commenting if he happ...</td>
      <td>hi7mdia</td>
      <td>qgo2dz</td>
      <td>Exofanjongdae</td>
      <td>60</td>
      <td>1.635311e+09</td>
    </tr>
  </tbody>
</table>
</div>



## 4. Extracting Sentiments using Text Blob

TextBlob is a Python library used for Natural Language Processing that allows us to determine the polarity and subjectivity of a given sentence. The polarity score ranges between -1 and 1, where -1 indicates a highly negative sentiment and 1 indicates a highly positive sentiment. The subjectivity score ranges from 0 to 1, with 0 representing objective or factual content and 1 representing subjective content.

In our approach, we will analyze sentiment on a per-sentence basis rather than per comment.

Sentences will be classified as follows:

Negative: polarity < 0
Positive: polarity > 0
Neutral: polarity = 0


```python
df_sentiment = df
df_sentiment.head()
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
      <th>comment</th>
      <th>comment_id</th>
      <th>parent_id</th>
      <th>username</th>
      <th>upvotes</th>
      <th>datetime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>honestly pretty happy with this - getting a pr...</td>
      <td>hi7j3kg</td>
      <td>qgo2dz</td>
      <td>eastsidegoondu</td>
      <td>192</td>
      <td>1.635309e+09</td>
    </tr>
    <tr>
      <th>1</th>
      <td>key part here is that it also unavailable for ...</td>
      <td>hi7hcbh</td>
      <td>qgo2dz</td>
      <td>MacWithoutCheese</td>
      <td>505</td>
      <td>1.635308e+09</td>
    </tr>
    <tr>
      <th>2</th>
      <td>great. we should stop treating public housing ...</td>
      <td>hi7i2xs</td>
      <td>qgo2dz</td>
      <td>pewsg</td>
      <td>314</td>
      <td>1.635308e+09</td>
    </tr>
    <tr>
      <th>3</th>
      <td>this is good news for those who really needs a...</td>
      <td>hi7ieqf</td>
      <td>qgo2dz</td>
      <td>PossibleConsistent77</td>
      <td>97</td>
      <td>1.635308e+09</td>
    </tr>
    <tr>
      <th>4</th>
      <td>never see noob hdb buyer commenting if he happ...</td>
      <td>hi7mdia</td>
      <td>qgo2dz</td>
      <td>Exofanjongdae</td>
      <td>60</td>
      <td>1.635311e+09</td>
    </tr>
  </tbody>
</table>
</div>




```python
#separate each comment into invididual sentences
df_sentiment["comment"] = [tokenize.sent_tokenize(item) for item in df_sentiment.comment]
```


```python
#split each sentence into individual rows
df_sentiment_1 = df_sentiment.explode("comment").reset_index(drop = True)
```


```python
df_sentiment_1.head()
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
      <th>comment</th>
      <th>comment_id</th>
      <th>parent_id</th>
      <th>username</th>
      <th>upvotes</th>
      <th>datetime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>honestly pretty happy with this - getting a pr...</td>
      <td>hi7j3kg</td>
      <td>qgo2dz</td>
      <td>eastsidegoondu</td>
      <td>192</td>
      <td>1.635309e+09</td>
    </tr>
    <tr>
      <th>1</th>
      <td>though i hope for your sakes you do not get sa...</td>
      <td>hi7j3kg</td>
      <td>qgo2dz</td>
      <td>eastsidegoondu</td>
      <td>192</td>
      <td>1.635309e+09</td>
    </tr>
    <tr>
      <th>2</th>
      <td>key part here is that it also unavailable for ...</td>
      <td>hi7hcbh</td>
      <td>qgo2dz</td>
      <td>MacWithoutCheese</td>
      <td>505</td>
      <td>1.635308e+09</td>
    </tr>
    <tr>
      <th>3</th>
      <td>and also the resale of the flat being be less ...</td>
      <td>hi7hcbh</td>
      <td>qgo2dz</td>
      <td>MacWithoutCheese</td>
      <td>505</td>
      <td>1.635308e+09</td>
    </tr>
    <tr>
      <th>4</th>
      <td>quite stringent and seems like exactly what ev...</td>
      <td>hi7hcbh</td>
      <td>qgo2dz</td>
      <td>MacWithoutCheese</td>
      <td>505</td>
      <td>1.635308e+09</td>
    </tr>
  </tbody>
</table>
</div>




```python
#define a function to obtain get polariy and subjectivity

def get_sentiment(text):
    blob = TextBlob(text)
    sentiment_polarity = blob.sentiment.polarity 
    sentiment_subjectivity = blob.sentiment.subjectivity 
    if sentiment_polarity > 0:
        sentiment_label = "positive"
    elif sentiment_polarity < 0:
        sentiment_label = "negative"
    else:
        sentiment_label = "neutral"
    #store result in a dictionary
    result = {"polarity": sentiment_polarity, 
             "subjectivity": sentiment_subjectivity,
             "sentiment": sentiment_label}
    return result    
```


```python
#apply function and create new column to store result
df_sentiment_1["sentiment_result"] = df_sentiment_1.comment.apply(get_sentiment)
```


```python
#split result (stored as dictionary) into individual key columns 
sentiment = pd.json_normalize(df_sentiment_1["sentiment_result"])
```


```python
#concatenate both dataframe together horizontally
df_1 = pd.concat([df_sentiment_1,sentiment], axis = 1)
df_1.head()
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
      <th>comment</th>
      <th>comment_id</th>
      <th>parent_id</th>
      <th>username</th>
      <th>upvotes</th>
      <th>datetime</th>
      <th>sentiment_result</th>
      <th>polarity</th>
      <th>subjectivity</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>honestly pretty happy with this - getting a pr...</td>
      <td>hi7j3kg</td>
      <td>qgo2dz</td>
      <td>eastsidegoondu</td>
      <td>192</td>
      <td>1.635309e+09</td>
      <td>{'polarity': 0.525, 'subjectivity': 1.0, 'sent...</td>
      <td>0.525000</td>
      <td>1.000000</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>1</th>
      <td>though i hope for your sakes you do not get sa...</td>
      <td>hi7j3kg</td>
      <td>qgo2dz</td>
      <td>eastsidegoondu</td>
      <td>192</td>
      <td>1.635309e+09</td>
      <td>{'polarity': 0.18333333333333335, 'subjectivit...</td>
      <td>0.183333</td>
      <td>0.633333</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2</th>
      <td>key part here is that it also unavailable for ...</td>
      <td>hi7hcbh</td>
      <td>qgo2dz</td>
      <td>MacWithoutCheese</td>
      <td>505</td>
      <td>1.635308e+09</td>
      <td>{'polarity': 0.0, 'subjectivity': 1.0, 'sentim...</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>3</th>
      <td>and also the resale of the flat being be less ...</td>
      <td>hi7hcbh</td>
      <td>qgo2dz</td>
      <td>MacWithoutCheese</td>
      <td>505</td>
      <td>1.635308e+09</td>
      <td>{'polarity': -0.09583333333333333, 'subjectivi...</td>
      <td>-0.095833</td>
      <td>0.095833</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>4</th>
      <td>quite stringent and seems like exactly what ev...</td>
      <td>hi7hcbh</td>
      <td>qgo2dz</td>
      <td>MacWithoutCheese</td>
      <td>505</td>
      <td>1.635308e+09</td>
      <td>{'polarity': 0.25, 'subjectivity': 0.25, 'sent...</td>
      <td>0.250000</td>
      <td>0.250000</td>
      <td>positive</td>
    </tr>
  </tbody>
</table>
</div>



## 4a. Sentiments, Polarity and Subjectivity Analysis

We are expecting most sentences to have a positive sentiment. This is true; however, the number of positive and neutral sentiment sentences are almost similar, with positive at 44% and neutral at 38%. This means the majority of the web-scraped content indicates that people either feel that the PLH model is a positive model to be rolled out or are sitting on the fence about this new model.


```python
plt.style.use("ggplot")

positive = len(df_1[df_1.sentiment == "positive"])
negative = len(df_1[df_1.sentiment == "negative"])
neutral = len(df_1[df_1.sentiment == "neutral"])

sentiment = [positive, neutral, negative]
sentiment_cat = ["positive", "neutral", "negative"]

sentiment.reverse()
sentiment_cat.reverse()

fig, ax = plt.subplots(figsize=(10,5))

palette = ["maroon", "darkslategrey", "seagreen"]

hbars = plt.barh(sentiment_cat, sentiment, color = palette, alpha = 0.5)

ax.bar_label(hbars, fmt='%.0f', color = "grey", padding = 5)

plt.xticks(np.arange(0,560,50).tolist())

plt.xlabel("Number of Comments")
plt.title("Overall Sentiment Distribution, 898 sentences", size = 13)
plt.show()
```


    
![png](output_65_0.png)
    


Web scraping comments are recorded exclusively on October 27th and 28th, 2021. The comment volume on October 28th, 2021, is minimal, making it more meaningful to analyze the hourly comment distribution for October 27th, 2021, instead.


```python
#converting date to appropriate dtypes
df_1["datetime"] = pd.to_datetime(df_1.datetime, unit = "s")
df_1["date"] = df_1["datetime"].dt.date
df_1["hour"] = df_1["datetime"].dt.hour
```


```python
df_date = df_1.groupby(["date", "sentiment"])["comment"].count().reset_index(name = "total_comment")
df_date.style.apply(lambda x: ['background: lightyellow' if x.total_comment < 40 \
                               else '' for i in x], axis=1)
```




<style type="text/css">
#T_55376_row3_col0, #T_55376_row3_col1, #T_55376_row3_col2, #T_55376_row4_col0, #T_55376_row4_col1, #T_55376_row4_col2, #T_55376_row5_col0, #T_55376_row5_col1, #T_55376_row5_col2 {
  background: lightyellow;
}
</style>
<table id="T_55376">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_55376_level0_col0" class="col_heading level0 col0" >date</th>
      <th id="T_55376_level0_col1" class="col_heading level0 col1" >sentiment</th>
      <th id="T_55376_level0_col2" class="col_heading level0 col2" >total_comment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_55376_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_55376_row0_col0" class="data row0 col0" >2021-10-27</td>
      <td id="T_55376_row0_col1" class="data row0 col1" >negative</td>
      <td id="T_55376_row0_col2" class="data row0 col2" >151</td>
    </tr>
    <tr>
      <th id="T_55376_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_55376_row1_col0" class="data row1 col0" >2021-10-27</td>
      <td id="T_55376_row1_col1" class="data row1 col1" >neutral</td>
      <td id="T_55376_row1_col2" class="data row1 col2" >324</td>
    </tr>
    <tr>
      <th id="T_55376_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_55376_row2_col0" class="data row2 col0" >2021-10-27</td>
      <td id="T_55376_row2_col1" class="data row2 col1" >positive</td>
      <td id="T_55376_row2_col2" class="data row2 col2" >379</td>
    </tr>
    <tr>
      <th id="T_55376_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_55376_row3_col0" class="data row3 col0" >2021-10-28</td>
      <td id="T_55376_row3_col1" class="data row3 col1" >negative</td>
      <td id="T_55376_row3_col2" class="data row3 col2" >7</td>
    </tr>
    <tr>
      <th id="T_55376_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_55376_row4_col0" class="data row4 col0" >2021-10-28</td>
      <td id="T_55376_row4_col1" class="data row4 col1" >neutral</td>
      <td id="T_55376_row4_col2" class="data row4 col2" >25</td>
    </tr>
    <tr>
      <th id="T_55376_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_55376_row5_col0" class="data row5 col0" >2021-10-28</td>
      <td id="T_55376_row5_col1" class="data row5 col1" >positive</td>
      <td id="T_55376_row5_col2" class="data row5 col2" >15</td>
    </tr>
  </tbody>
</table>




As anticipated, the volume of comments reaches its highest point at 5 a.m., then gradually declines throughout the day as the topic on PLH loses traction. Both positive and negative sentiment comments follow a similar pattern. In contrast, neutral sentiment comments diverge from this trend, displaying an inverse relationship to positive and negative sentiment comments as the day progresses.


```python
#create dataframe of all dates and time
a = np.arange(0,25,1).tolist()
df_total_hour = pd.DataFrame(a, columns = ["hour"])
df_total_date = pd.DataFrame([datetime.date(2021, 10, 27), datetime.date(2021, 10, 28)], columns = ["date"])
df_total_hour_date = pd.merge(df_total_date, df_total_hour, how = "cross")

#positive
positive = df_1[df_1.sentiment == "positive"]
positive_1 = pd.merge(positive, df_total_hour_date, how = "outer", on = ["date", "hour"])
positive_date = positive_1.groupby(["date", "hour"])["comment"].count().reset_index(name = "total_comment")
positive_date27 = positive_date[positive_date.date == datetime.date(2021, 10, 27)]

#negative
negative = df_1[df_1.sentiment == "negative"]
negative_1 = pd.merge(negative, df_total_hour_date, how = "outer", on = ["date", "hour"])
negative_date = negative_1.groupby(["date", "hour"])["comment"].count().reset_index(name = "total_comment")
negative_date27 = negative_date[negative_date.date == datetime.date(2021, 10, 27)]

#neutral
neutral = df_1[df_1.sentiment == "neutral"]
neutral_1 = pd.merge(neutral, df_total_hour_date, how = "outer", on = ["date", "hour"])
neutral_date = neutral_1.groupby(["date", "hour"])["comment"].count().reset_index(name = "total_comment")
neutral_date27 = neutral_date[neutral_date.date == datetime.date(2021, 10, 27)]

fig, (ax1, ax2, ax3) = plt.subplots(nrows = 3, ncols = 1, sharex = True, sharey = True, figsize=(10,8))

#ax1 positive and negative
ax1.plot(positive_date27.hour, positive_date27.total_comment, label = "positive", color = "seagreen", marker = "o", markersize = 3)
ax1.plot(negative_date27.hour, negative_date27.total_comment, label = "negative", color = "crimson", marker = "o", markersize = 3)
ax1.fill_between(positive_date27.hour, positive_date27.total_comment, negative_date27.total_comment, color = "seagreen",
                alpha = 0.5)

#ax2 positive and neutral
ax2.plot(positive_date27.hour, positive_date27.total_comment, label = "positive", color = "seagreen", marker = "o", markersize = 3)
ax2.plot(neutral_date27.hour, neutral_date27.total_comment, label = "neutral", color = "darkslategrey", alpha = 0.5, marker = "o", markersize = 3)
ax2.fill_between(positive_date27.hour, positive_date27.total_comment, neutral_date27.total_comment,
                 positive_date27.total_comment > neutral_date27.total_comment, interpolate = True, color = "seagreen",
                alpha = 0.5)
ax2.fill_between(positive_date27.hour, positive_date27.total_comment, neutral_date27.total_comment,
                 positive_date27.total_comment < neutral_date27.total_comment, interpolate = True, color = "darkslategrey",
                alpha = 0.5)

#ax3 neutral and negative
ax3.plot(negative_date27.hour, negative_date27.total_comment, label = "negative", color = "crimson",  marker = "o", markersize = 3)
ax3.plot(neutral_date27.hour, neutral_date27.total_comment, label = "neutral", color = "darkslategrey", alpha = 0.5,  marker = "o", markersize = 3)
ax3.fill_between(negative_date27.hour, negative_date27.total_comment, neutral_date27.total_comment,
                 negative_date27.total_comment > neutral_date27.total_comment, interpolate = True, color = "crimson",
                alpha = 0.5)
ax3.fill_between(negative_date27.hour, negative_date27.total_comment, neutral_date27.total_comment,
                 negative_date27.total_comment < neutral_date27.total_comment, interpolate = True, color = "darkslategrey",
                alpha = 0.5)


plt.xticks(np.arange(0,25,1).tolist())
plt.yticks(np.arange(0,150,20).tolist())

ax1.legend()
ax2.legend()
ax3.legend()
    
plt.legend()

# plt.plot(positive_date28.hour, positive_date28.total_comment)
plt.suptitle("Total Sentences Distribution on 27th October 2021", size = 13)
plt.tight_layout()
plt.show()
```


    
![png](output_70_0.png)
    


Overall, the sentiment tends to be positive on a per-sentence basis, with most positive remarks exhibiting a polarity around +0.25. In contrast, sentences expressing negative sentiment hover closer to zero. Referring to the earlier bar chart, despite the higher count of positive sentiment sentences, the overall average polarity is 0.0875. However, this figure is calculated per sentence, and to obtain a more comprehensive understanding, we need to consider the average polarity across all sentences within each comment.


```python
#sentiment for each sentence
df_sentence_summary = df_1[["comment", "polarity", "subjectivity"]]
df_sentence_summary.describe().style.apply(lambda x: ['background: lightyellow' if x.polarity > 0.08 \
                                                     and x.polarity < 0.09 else '' for i in x], axis=1)
```




<style type="text/css">
#T_e2a10_row1_col0, #T_e2a10_row1_col1 {
  background: lightyellow;
}
</style>
<table id="T_e2a10">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_e2a10_level0_col0" class="col_heading level0 col0" >polarity</th>
      <th id="T_e2a10_level0_col1" class="col_heading level0 col1" >subjectivity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_e2a10_level0_row0" class="row_heading level0 row0" >count</th>
      <td id="T_e2a10_row0_col0" class="data row0 col0" >901.000000</td>
      <td id="T_e2a10_row0_col1" class="data row0 col1" >901.000000</td>
    </tr>
    <tr>
      <th id="T_e2a10_level0_row1" class="row_heading level0 row1" >mean</th>
      <td id="T_e2a10_row1_col0" class="data row1 col0" >0.087135</td>
      <td id="T_e2a10_row1_col1" class="data row1 col1" >0.353105</td>
    </tr>
    <tr>
      <th id="T_e2a10_level0_row2" class="row_heading level0 row2" >std</th>
      <td id="T_e2a10_row2_col0" class="data row2 col0" >0.232180</td>
      <td id="T_e2a10_row2_col1" class="data row2 col1" >0.308896</td>
    </tr>
    <tr>
      <th id="T_e2a10_level0_row3" class="row_heading level0 row3" >min</th>
      <td id="T_e2a10_row3_col0" class="data row3 col0" >-1.000000</td>
      <td id="T_e2a10_row3_col1" class="data row3 col1" >0.000000</td>
    </tr>
    <tr>
      <th id="T_e2a10_level0_row4" class="row_heading level0 row4" >25%</th>
      <td id="T_e2a10_row4_col0" class="data row4 col0" >0.000000</td>
      <td id="T_e2a10_row4_col1" class="data row4 col1" >0.000000</td>
    </tr>
    <tr>
      <th id="T_e2a10_level0_row5" class="row_heading level0 row5" >50%</th>
      <td id="T_e2a10_row5_col0" class="data row5 col0" >0.000000</td>
      <td id="T_e2a10_row5_col1" class="data row5 col1" >0.337500</td>
    </tr>
    <tr>
      <th id="T_e2a10_level0_row6" class="row_heading level0 row6" >75%</th>
      <td id="T_e2a10_row6_col0" class="data row6 col0" >0.200000</td>
      <td id="T_e2a10_row6_col1" class="data row6 col1" >0.554167</td>
    </tr>
    <tr>
      <th id="T_e2a10_level0_row7" class="row_heading level0 row7" >max</th>
      <td id="T_e2a10_row7_col0" class="data row7 col0" >1.000000</td>
      <td id="T_e2a10_row7_col1" class="data row7 col1" >1.000000</td>
    </tr>
  </tbody>
</table>





```python
g = sns.jointplot(data = df_sentence_summary, 
           x = "polarity", y = "subjectivity",
        kind = "scatter", alpha = 0.5)

g.ax_marg_x.set_xlim(-1.02, 1.06)
g.ax_marg_y.set_ylim(-0.02, 1.06)

plt.suptitle("Comments Subjectivity and Polarity Distribution", y = 1.02, x = 0.47)
plt.show()
```

    /opt/anaconda3/lib/python3.11/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/anaconda3/lib/python3.11/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    


    
![png](output_73_1.png)
    


It is crucial to consider the tally of upvotes and downvotes, as these reflect the sentiments of users with similar viewpoints who did not leave comments. The average polarity value stands at 0.09, indicating a near-neutral tone. In general, the comments within this thread lean towards neutrality, albeit with a slight positive inclination.

## 5. Natural Language Processing (NLP)

To gain insights into the main topics that interest or concern web scraping enthusiasts, it's essential to analyze the top 10 words. Moreover, understanding the part-of-speech and grammatical category of each word, whether it's a verb/action or noun, can provide further insights into potential topics. To extract the overall top 10 words, we'll follow these steps:

Tokenization: Break down each sentence into individual words.
POS-Tagging: Categorize each word into its grammatical category.
Stemming: Reduce each word to its root base (e.g., "people" becomes "peopl", "running" becomes "run").
Filtering Stop Words: Remove words that don't contribute additional meaning to the sentence (e.g., "me", "a", "the").


```python
#selecting relevant attributes/ columns 
df_nlp = df_1[["comment", "sentiment"]]
df_nlp.head()
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
      <th>comment</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>honestly pretty happy with this - getting a pr...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>1</th>
      <td>though i hope for your sakes you do not get sa...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2</th>
      <td>key part here is that it also unavailable for ...</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>3</th>
      <td>and also the resale of the flat being be less ...</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>4</th>
      <td>quite stringent and seems like exactly what ev...</td>
      <td>positive</td>
    </tr>
  </tbody>
</table>
</div>




```python
#tokenize
df_nlp["comment"] = [word_tokenize(i) for i in df_nlp.comment]
df_nlp.head()
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
      <th>comment</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[honestly, pretty, happy, with, this, -, getti...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[though, i, hope, for, your, sakes, you, do, n...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[key, part, here, is, that, it, also, unavaila...</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[and, also, the, resale, of, the, flat, being,...</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[quite, stringent, and, seems, like, exactly, ...</td>
      <td>positive</td>
    </tr>
  </tbody>
</table>
</div>




```python
#remove white/ blanks
df_nlp["comment"] = [[i for i in item if i != ""] for item in df_nlp.comment]
df_nlp.head()
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
      <th>comment</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[honestly, pretty, happy, with, this, -, getti...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[though, i, hope, for, your, sakes, you, do, n...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[key, part, here, is, that, it, also, unavaila...</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[and, also, the, resale, of, the, flat, being,...</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[quite, stringent, and, seems, like, exactly, ...</td>
      <td>positive</td>
    </tr>
  </tbody>
</table>
</div>




```python
#process sequence of words using pos_tag()
df_nlp["comment"] = [nltk.pos_tag(item) for item in df_nlp.comment]
df_nlp.head()
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
      <th>comment</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[(honestly, RB), (pretty, RB), (happy, JJ), (w...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[(though, IN), (i, JJ), (hope, VBP), (for, IN)...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[(key, JJ), (part, NN), (here, RB), (is, VBZ),...</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[(and, CC), (also, RB), (the, DT), (resale, NN...</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[(quite, RB), (stringent, JJ), (and, CC), (see...</td>
      <td>positive</td>
    </tr>
  </tbody>
</table>
</div>




```python
#stemming 
s_stemmer = SnowballStemmer(language='english')

df_nlp["comment_1"] = [[s_stemmer.stem(i[0]) for i in item] for item in df_nlp.comment]
df_nlp["comment_2"] = [[i[1] for i in item] for item in df_nlp.comment]

df_nlp["comment_1"] = [[re.sub(r"[^\w']", '', i) for i in item] for item in df_nlp.comment_1]
df_nlp["comment_2"] = [[re.sub(r"[^\w']", '', i) for i in item] for item in df_nlp.comment_2]
df_nlp.head()
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
      <th>comment</th>
      <th>sentiment</th>
      <th>comment_1</th>
      <th>comment_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[(honestly, RB), (pretty, RB), (happy, JJ), (w...</td>
      <td>positive</td>
      <td>[honest, pretti, happi, with, this, , get, a, ...</td>
      <td>[RB, RB, JJ, IN, DT, , VBG, DT, JJ, NN, NN, MD...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[(though, IN), (i, JJ), (hope, VBP), (for, IN)...</td>
      <td>positive</td>
      <td>[though, i, hope, for, your, sake, you, do, no...</td>
      <td>[IN, JJ, VBP, IN, PRP, NNS, PRP, VBP, RB, VB, ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[(key, JJ), (part, NN), (here, RB), (is, VBZ),...</td>
      <td>neutral</td>
      <td>[key, part, here, is, that, it, also, unavail,...</td>
      <td>[JJ, NN, RB, VBZ, IN, PRP, RB, JJ, IN, JJ, RB,...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[(and, CC), (also, RB), (the, DT), (resale, NN...</td>
      <td>negative</td>
      <td>[and, also, the, resal, of, the, flat, be, be,...</td>
      <td>[CC, RB, DT, NN, IN, DT, JJ, VBG, VB, RBR, JJ, ]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[(quite, RB), (stringent, JJ), (and, CC), (see...</td>
      <td>positive</td>
      <td>[quit, stringent, and, seem, like, exact, what...</td>
      <td>[RB, JJ, CC, VBZ, IN, RB, WP, NN, RB, VBD, VBG...</td>
    </tr>
  </tbody>
</table>
</div>




```python
#separate each item into individual columns
df_nlp_1 = df_nlp[["sentiment", "comment_1", "comment_2"]].explode(["comment_1", "comment_2"]).explode(["comment_1", "comment_2"])
df_nlp_1.head()
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
      <th>sentiment</th>
      <th>comment_1</th>
      <th>comment_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>positive</td>
      <td>honest</td>
      <td>RB</td>
    </tr>
    <tr>
      <th>0</th>
      <td>positive</td>
      <td>pretti</td>
      <td>RB</td>
    </tr>
    <tr>
      <th>0</th>
      <td>positive</td>
      <td>happi</td>
      <td>JJ</td>
    </tr>
    <tr>
      <th>0</th>
      <td>positive</td>
      <td>with</td>
      <td>IN</td>
    </tr>
    <tr>
      <th>0</th>
      <td>positive</td>
      <td>this</td>
      <td>DT</td>
    </tr>
  </tbody>
</table>
</div>




```python
#remove all stop words 
stop_words_lst = list(nlp.Defaults.stop_words)
df_nlp_1 = df_nlp_1[~df_nlp_1.comment_1.isin(stop_words_lst)]
```

## 5a. Top 10 Most Occuring Word


```python
#grouping and counting total number of occurance for each word 
df_nlp_2 = df_nlp_1[df_nlp_1.comment_1 != ""]

df_nlp_2 = df_nlp_2.groupby(["sentiment","comment_1", "comment_2"])["comment_1"].count().reset_index(name = "total_number")
df_nlp_2.sort_values(["sentiment", "total_number"], ascending = [False, False], inplace = True)

df_nlp_2.head()
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
      <th>sentiment</th>
      <th>comment_1</th>
      <th>comment_2</th>
      <th>total_number</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2678</th>
      <td>positive</td>
      <td>peopl</td>
      <td>NNS</td>
      <td>55</td>
    </tr>
    <tr>
      <th>2215</th>
      <td>positive</td>
      <td>flat</td>
      <td>NNS</td>
      <td>44</td>
    </tr>
    <tr>
      <th>2334</th>
      <td>positive</td>
      <td>hous</td>
      <td>NN</td>
      <td>35</td>
    </tr>
    <tr>
      <th>2266</th>
      <td>positive</td>
      <td>good</td>
      <td>JJ</td>
      <td>31</td>
    </tr>
    <tr>
      <th>2301</th>
      <td>positive</td>
      <td>hdb</td>
      <td>NN</td>
      <td>31</td>
    </tr>
  </tbody>
</table>
</div>




```python
#select top 10 words from each sentiment category 
df_nlp_15 = df_nlp_2.groupby("sentiment").head(10)
df_nlp_15.reset_index(inplace = True)
df_nlp_15["comment_1_2"] = df_nlp_15["comment_1"] + " - " + df_nlp_15["comment_2"]
df_nlp_15.head()
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
      <th>index</th>
      <th>sentiment</th>
      <th>comment_1</th>
      <th>comment_2</th>
      <th>total_number</th>
      <th>comment_1_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2678</td>
      <td>positive</td>
      <td>peopl</td>
      <td>NNS</td>
      <td>55</td>
      <td>peopl - NNS</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2215</td>
      <td>positive</td>
      <td>flat</td>
      <td>NNS</td>
      <td>44</td>
      <td>flat - NNS</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2334</td>
      <td>positive</td>
      <td>hous</td>
      <td>NN</td>
      <td>35</td>
      <td>hous - NN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2266</td>
      <td>positive</td>
      <td>good</td>
      <td>JJ</td>
      <td>31</td>
      <td>good - JJ</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2301</td>
      <td>positive</td>
      <td>hdb</td>
      <td>NN</td>
      <td>31</td>
      <td>hdb - NN</td>
    </tr>
  </tbody>
</table>
</div>




```python
g = sns.catplot(data = df_nlp_15,
            x = "total_number", y = "comment_1_2",
            kind = "bar",
            hue = "comment_2",
            col = "sentiment",
            dodge = False,
            sharey = False,
            alpha = 0.8, height = 6, aspect = 1)

for ax in g.axes.ravel():
    for c in ax.containers:
        labels = [f'{v.get_width()}' for v in c]
        ax.bar_label(c, labels=labels, label_type='edge', color = "grey", size = 15)

title = ["positive", "neutral", "negative"]

axes = g.axes.flatten()

for i in range(3):
    axes[i].set_title(title[i], size = 15)
    l = axes[i].get_yticklabels()
    axes[i].set_yticklabels(l, fontsize=15)
    axes[i].set_ylabel("")
    axes[i].set_xlabel("Total Number of Words", size = 10)

new_title = 'POS Tag'
g._legend.set_title(new_title)

new_labels = ['NNS: common noun, plural', 'NN: common noun', 'JJ: Adjective', 'RB: adverb', 'VB: Verb', 'VBZ: 3rd Person', 'IN: Preposition']
for t, l in zip(g._legend.texts, new_labels):
    t.set_text(l)

g._legend.set_bbox_to_anchor((1.12, 0.77))

plt.suptitle("Top 10 Most Common Words", size = 20, x = 0.53)

plt.tight_layout()
plt.show()

```


    
![png](output_87_0.png)
    


## 6. Summary

Overall, the opinions about the Prime Housing Location are not overwhelmingly positive; they tend towards neutrality.

It's important to note that while using TextBlob for sentiment analysis, the library interprets sentences at face value. Ambiguities such as sarcasm, cynicism, and emotional undertones are not captured. For instance, a sentence like "let us see if this model really works in the next few years" could have a hidden negative connotation (e.g., disbelief, cynicism), but TextBlob returns a sentiment value of 0.

Additionally, the intensity conveyed by slangs or emojis might be lost during conversion. A laughing-with-tears emoji could represent happiness or disbelief. The intensity of Singlish terms such as "shiok" or even "kbpb" might be diminished when converted to proper English, like "good" and "complain." However, using TextBlob for sentiment analysis is generally still acceptable.

The top 10 words are indicative of the public's concerns (e.g., resale, flat, house, BTO, income, prime, HDB, singles). Interestingly, the word "single" appeared 29 times in the negative category, more than the 25 times in the positive category.
