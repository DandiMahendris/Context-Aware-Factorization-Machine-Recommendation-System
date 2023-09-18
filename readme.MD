<a name="readme-top"></a>

<!-- TABLE OF CONTENTS -->

<summary><b>Table of Contents</b></summary>
<ol>
  <li><a href="#about-the-project">About The Project</a></li>
  <li><a href="#business-objective">Business Objective</a></li>
  <li><a href="#business-metrics">Business Metrics</a></li>
  <li><a href="#getting-started">Getting Started</a></li>
  <li>
    <a href="#data-workflow">Data Workflow</a>
    <ul>
      <li><a href="#dataset">Dataset</a></li>
      <li><a href="#data-preparation">Data Preparation</a></li>
      <li><a href="#data-preprocessing-and-feature-engineering">Data Preprocessing and Feature Engineering</a></li>
      <li><a href="#data-modelling">Data Modelling</a></li>      
    </ul>
  </li>
  <li>
    <a href="#prediction-using-api-and-streamlit">Prediction using API and Streamlit</a>
    <ul>
      <li><a href="#how-to-run-by-api?">How To Run by API?</a></li>
      <li><a href="#data-input">Data Input</a></li>
    </ul>
  </li>
</ol>

<!-- About the Project -->
# Recommendation System

Steam platform is the biggest video game digital distribution service for PC Gaming nowadays under Valve Corporartion. According to annual statistics report in 2022, steam reached peak of online user count of 33 millions with impressive number of game downloads up to 44.7 billion gigabytes downloaded over the year. One of the key reasons Steam is growing so rapidly is the good search-ability in store, which was mentioned in the report as well. They are working on a new recommendation system driven by machine learning to find games that match the player’s personal preferences. Although the algorithm is just part of the search-ability solution, they are also building more live and appreciation features and continually evaluating the overall design of the store.

On the other hand, the recommender system also affected by Matthew Effect, which means those games developed by big game developing companies receive more budget on advertisement so that they can be very popular. Popular games will appear on the top position on the store web pages and attract more users to buy including you and your friends. Meanwhile, games developed by small studios or individual developers are not that lucky. Some midle-budget user is more likely to wait their own preference games that will be launched on few month ahead or just simply waiting their preference games to get discount. It can leads this typical user to churn or inactive in some periods meanwhile theres should be bunch of hidden gems that matched their preference based on user activity log or given rating on several games.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

# Business Objective

Our goal is to create a recommendation system approach for product-suggestions focusing on those middle-user that which games they will buy based on their given rating or preference (activity log) that more likely to be a hidden gems like either lowest price product or untrending games that comes out not from big games developer.

- First method is **Content-based recommendation system**, is to generate more similar games but, focusing in **novelty** (how unknown user to related item) and **serependity** (how related the item to user) to get less trending games.

- Second, is **Collaborative Filtering** with **Context-Aware Factorization Machine** for price rate and release date of games to give recommendation based on rating given (*Regression*) and do user recommend it or not (*Classification*).

- Third, is **Collaborative Filtering** with **Bayesian Pesonalized Ranking** and **Alternating Least Square** to utilize the implicit data (hours play) by user.

The dataset will come out in three parts, first one include User IDs, Game IDs, Rating (*Explicit*), the hours they had played (*Implicit*) and do they recommended it or not. Second part is content based likely consist the game's title, genre, description, release date and price-range. These dataset is retrieved from Kaggle Dataset of [**Steam's user log**](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam) and [**Steam's game description**](https://www.kaggle.com/datasets/trolukovich/steam-games-complete-dataset) which are updated until 2022-12-31.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

# Business Metrics

We evaluate the accuracy of our results by separating the dataset into a test set and a training set. Specifically, we attempt to predict the values in the test set using the training set. 

The evaluation metrics that will be used in this recommendation system approach is **nDCG** for *N-Ranking recommendation* and **RMSE** for *Rating-based prediction*.

# Dataset

**Steam's User log** is consist total **[37.800.000, 23]** and **steam games description** is consist total of **[40.000, 20]**. However for 37.8 Mio in total instance will greatly cost computational time in training the recommendation system, hence we reduce to used only **1000_000** total instances. Dataset is splitted using `train_test_split` library from `sklearn.model_selection` package to produce equal distributed features from the original data.

<style>
    table {
        font-size: 10px; /* You can adjust the font size as needed */
        width: 80%; /* Adjust the table width as needed */
    }
</style>

| Column Name        | Describe                            | dtype    |
| :----------------- | :---------------------------------- | :------- |
| *user_id*          | User unique identifier              | object   |
| *app_id*           | Steam's game unique identifier      | object   |
| *date*             | Steam's game release date           | datetime |
| *is_recommended*   | User vote up the game if True       | bool     |
| *hours*            | Hour of user spent on game          | float64  |
| *title*            | Steam games name                    | object   |
| *rating*           | Rating given by user on played game | object   |
| *price_final*      | price after discounted              | float64  |
| *developer*        | Steam's game developer              | object   |
| *publisher*        | Steam's game publisher              | object   |
| *Categories*       | Steam's game category               | object   |
| *genre*            | Steam's game genre                  | object   |
| *popular_tags*     | Steam's game popular tags           | object   |
| *game_description* | Steam's game description            | object   |
|                    |                                     |          |

<p align=left>
<img src="pics/0.png" width=80%>
</p>

# Methods 

There are several datasets setting that we are used in this project to observed several Recommendation System model performance.

1. *Table 1: User IDs, App IDs, title, categories, genre, popular tags, and game_description*. <br>
This table is used to utilize the **content-based recommendation system** by ignoring user preference, and just calculate similarity between games based on game's title, genre, tags and description (App IDs). Dimension of data used is 1000_000 x 7, and will be increase by being added Interaction counts, Novelty scores, Serependity scores features to calculate those feature effect to recommendation.

2. *Table 2: User IDs, App IDs, Rating*.<br>
This table is only contains user_id, app_id and rating given by user to certain games. The ratings given will be classified into certain ordinal class, and utility matrix is given by these features. Dimension of data used is highly reduced to 20_000 x 3, since it will harm computational cost we will split using [`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) from `sklearn.model_selection` libraries. **Neighborhood Collaborative Filtering** is applied to this dataset.

3. *Table 3: User IDs, App IDs, Rating, price_final, release date*.<br>
This table contains price_final and release date to utilize context-aware recommendation later on this project, to utilize user preference based on its price or released date. Dimension data instances used is same with *Table 2* with 2 more features. The prediction model use **Context-Aware Regression Factorization Machines** to predict rating given by user based on game's price and release date context.

4. *Table 4: User IDs, App IDs, is_recommended, price_final*.<br>
This table is same with *Table 3* but the difference is rather than use continous data like rating, we are using binary data like is_recommended to predict whether user-u will recommend the item-i or not. The prediction model use **Context-Aware Classification Factorization Machines** to predict whether user recommend this games or not based on game's price and release date context.

5. *Table 5: User IDs, App IDs, hours*.<br>
This table is same with *Table 3* but rather than use explicit data like ratings, we are use implicit data like hours play of user-u on item-i. The prediction model use **Bayesian Personalized Ranking Model** from [`implicit`](https://pypi.org/project/implicit/) libraries to predict whether user will be played this games or not.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

# Exploratory Data Analysis

### Ratings

Rating feature on steam dataset will be our target on several recommendation system model as explicit data. Next, these categorical rating will be converted into numerical rating scale by 1-9 as ordered from the lowest to highest below.

<p align=center>
<img src="pics/image-1.png" width=20%>
</p>

<p align=center>
<img src="pics/newplot.png" width=90%>
</p>

The majority of ratings fall into the positive category, with "Very Positive" and "Overwhelmingly Positive" being the two highest-rated categories by a significant margin. "Very Positive" is given 574,438 times by users to several games they interact with as the biggest number of rating given in steam dataset, followed by "Overwhelmingly Positive" is given by 262,271 times. This suggests that games in steam has received a largely positive response from users. 

While positive ratings dominate, there is still a substantial number of users who have given a "Mixed" rating. This indicates that there is some variability in how users perceive or experience the games, with both positive and negative feedback. 

The "Very Negative" rating has the fewest counts that given 92 times by users in steam for few games, however negative rating given by user is more likely low for several games like "Overwhelmingly Positive" is given 191 times and "Negative" only given by 128 times. This is a relatively positive sign, as it suggests that the games in steam is not widely disliked. 

### Price Rates

<p align=center>
<img src="pcis/newplot-2.png" width=70%>
</p>

Since we are intended to use price as context to define what rating will user given to certain item based on its price range. These price range is subjective and in this projcet are separated into 5 categories. 

- $0 - $7.5 (Free - IDR 120,000) as **cheap** games. <br>
Free games is not casually considered high for certain user and most likely offer buy-in-game (skins, characters) such as Fornite, Valorant, Genshin Impacts. under $7.5 (IDR 120.000) games is considered as best affordable game's price and most likely some good games is reduced by discount into this category. Most users is would buy game in this price range either because its affordable or only for waiting big games that would come out later.

- $7.5 - $22.5 (IDR 120,000 - IDR 350,000) as **medium** games. <br>
This price range is considered as good game's price, sometimes big games will be discounted into this price range. The big trending games over 3 years release date also fall into this price range.

- $23 - $33 (IDR 350,000 - IDR 500,000) as **expensive** games. <br>
Currently, more newest games is more likely fall into this price range and also the biggest game price might come into this price range after discount.

- $33 - $53 (IDR 500,000 - IDR 800,000) as **luxury** games. <br>
Trending or newest AAA games is most likely come into this price range, but price range should not be considered if user want to experience luxury trending games.

- $53 - ~ (IDR 800,000) as **overpriced** games.<br>
This price range consider as the most expensive game's price on steams and only few users come to afford.

### Release date range

<p align=center>
<img src="pics/newplot-3.png" width=70%>
</p>

Release date is also will be considered to use in defined predicted rating that user would give. Some user is more likely not prefer over 10 years released date games and consider more into new stuff. Release date range are separated into 5 ranges:

- 0 - 1 years as **newest**.<br>
Some games that released on this range more likely have high prices and lower interactions, but low prices game with more high rating given is considered as good stuff.

- 1 - 3 years as **new**.<br>
Few games start to get some discounts when comes into this ranges and consider as most-waiting games that user want to buy. Games on this range will have more interactions count rather than above.

- 3 - 5 years as **mid**.<br>
Big AAA games would fall into good prices after come into this range.

- 5 - 10 years as **old**.<br>
Some user still want to play this games range and its price is drastically reduced, so this range is most affordable games.

- over 10 years as **outdated**.<br>
User will have low preference to outdated games.

# Preprocessing and Featuring Data

<p align = center>
<img src="https://miro.medium.com/v2/resize:fit:1064/1*mz9tzP1LjPBhmiWXeHyQkQ.png" width=60%>
</p>

# 1. Content-Based Recommendation System

### Text Preprocessing: Tokenization

<p align = center>
<img src="https://miro.medium.com/v2/resize:fit:1050/0*EKgminT7W-0R4Iae.png" width=30%>
</p>

Tokenization is the process of converting sensitive data into non-sensitive data called tokens.

Input: “Keep your eyes on this one, because it’s one quality Action RPG”

Output: [‘Keep’, ‘your’, ‘eyes’, ‘on’, ‘this’, ‘one’, ‘because’, ‘its’, ‘one’, ‘quality’, ‘action’, ‘rpg’]

In this example, the input text is tokenized into a list of individual words or tokens. The punctuation marks are removed during the tokenization proces.

In data preprocessing, tokenization is assigned to features names `tags` that combined from game title, game genre, game popular tags, and game description. However we ignore the game publisher and game developer since we do not want to get recommendation from similar developer or publisher.

```python
# tokenize the tags information and lowerize the character
new_data_content['tags'] = new_data_content['tags'].str.replace(',',' ')
new_data_content['tags'] = new_data_content['tags'].apply(lambda x: x.lower() if isinstance(x, str) else np.nan)
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Text Preprocessing: Stemming

<p align=center>
<img src="https://cdn-images-1.medium.com/v2/resize:fit:1024/format:png/1*I8vq4bEnlNZdhQ_u0nUTng.png" width=30%>
</p>

Stemming is the process of reducing derived words to their base word form.<br>

For example, a stemmer for English operating on the stem “cat” should identify such strings as “cats”, “catlike”, and "catty". <br>
A stemming algorithm might also reduce the words “fishing”, “fished”, and “fisher” to the stem "fish".

`SnowballStemmer` from `nltk.stem` package is used in this process

```python
from nltk.stem import SnowballStemmer

ss = SnowballStemmer(language='english')

#defining the stemming function
def stem(text):
    y=[]
    
    for i in text.split():
        y.append(ss.stem(i))
    
    return " ".join(y)
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Text Preprocessing: Term-Frequency (TF) and Inverse Document Frequency (IDF) Vectorizer

**Term-Frequency** is giving each word weight/value based on their occurence, rather than perform One Hot Encoder to split a word into a new feature.

$$ TF(word,document) = \frac{Number \ of \ word \ appear \ in \ a \ document}{Total \ word \ in \ a \ document} $$

However some frequent word can put into a sentence, hence term-frequency becoming bigger compared to other, and need to adjust to get better added value. **Inverse Document Frequency** can penalized the TF Score, means that if a term appears in many documents, its IDF value will decrease.

$$ IDF(word, document) = \log({\frac{Total \ Number \ of \ Document \ in \ corpus}{Number \ of \ Documents \ with \ word}}) $$

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

count_matrix = vectorizer.fit_transform(new_data_content['tags'])
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Cosine Similarity

Cosine similarity measure the direction between the two items or word. The cosine similarity between two vectors is measured in ‘θ’.

- If θ = 0°, the ‘x’ and ‘y’ vectors overlap, thus proving they are similar.<br>
- If θ = 90°, the ‘x’ and ‘y’ vectors are dissimilar.

$$ \cos{\theta{}} = \frac{A.B}{\lVert A \rVert  \ \lVert B \rVert} $$

<p align=center>
<img src="pics/2.png">
</p>

```python
from sklearn.metrics.pairwise import cosine_similarity

# Calculate cosine similarity toward tags feature after stemmed
cosine_sim = cosine_similarity(count_matrix, count_matrix)
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Content Similar Recommendation

Product-suggestion is performed based on the similar app_id or steam games, that will return n-games that similar of played or bought item by user log.

```python
def item_recommendations(id, cosine_sim=cosine_sim):
    indices = pd.Series(new_data_content.index, index = new_data_content['app_id']).drop_duplicates() 
    idx = indices[id]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Extract the app_ids from sim_scores[1:11]
    app_ids_to_select = [sim[0] for sim in sim_scores[1:11]]

    recommendation = new_data_content.loc[new_data_content.app_id.isin(app_ids_to_select)]

    return recommendation
```
<p align=left>
<img src="pics/3.png" width=60%>
</p>
<p align="right">(<a href="#readme-top">back to top</a>)</p>

The games we searched for similarity is Assassin's Creed Odyssey, and its found out the content-based recommendation for using similarity function only return most similar title games like another Assassin's Creed game fanchise above being the highest similarity. However, this recommendation system could be better for user who prefer franchise games.

### Novelty

Novelty measure how unknown the items by all user in our dataset.

$$ Novelty(item_{i}) = \frac{count(user \ who \ has \ not \ interacted \ with \ item \ i)}{count(all \ user)}  $$

- Novelty ~= 0, items is well known by all users
- Novelty ~= 1, items is unknown by all users

```python
def calculate_novelty_based_on_rarity(app_data):
    # Assuming 'interaction_count' is a column in your dataset representing how often the app has been interacted with.
    
    app_data['novelty_score'] = (app_data.user_id.nunique() - app_data['interactions_count']) / app_data.user_id.nunique()
    
    return app_data
```

#### Novelty and Similar Recommendation

<p align=left>
<img src="pics/4.png" width=70%>
</p>

Recommendation system with novelty score considered is indeed give more better result over previous recommendation. It give similarity game genre and tags (Open-world, RPG) toward Assassin's Creed Odyssey like Ghost Recon Wildlands, Dishonered, Far Cry 5, etc. However some game recommendation turn out to be less similarity like FrostPunk which considered as strategy builder games but it might have some similar tags with our games. This issue can be resolved by added more related genre or tags and remove unrelated tags in tokenization process.

Novelty score can produce high score since number of unique user is in biggest number, in this case total user is 10266 in total. However, the novelty can become a bias since more recent games (**new released game**) **have lower interactions_count**, e.g Far Cry 5, Watch Dogs 2
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Serependity

Serependity refers to ability to provide recommendation that are both surprising and useful. These are not only novel and diverse like previous, but also relevant to the user's interest. It's challenging to create a precise function for serendipity because it depends on various factors and is highly subjective. However, you can create a simple Python function to calculate a basic serendipity score based on some criteria.

Serependity is simply combination of **Relevance** of items and **Unexpectedness** of item.

$$ Serependity(item_{i} = Relevance(item_{i}) * Unexpectedness(item_{i})) $$

Unexpectedness refer to the recommendation is not similar item to past recommendation

$$ Unexpectedness(i) = \frac{\sum_{hEH} (1-Similarity(i,h))}{Number \ of \ H} $$

while i = item in user recommendation<br>
&emsp;&emsp; H = Historical item interaction list(such as rating activity, click, etc)

#### Serependity and Similar Recommendation

<p align=left>
<img src="pics/5.png" width=70%>
</p>

Serependity score based recommendation return more randomize games that similar to our searched games (Assassin's Creed: Odyssey) but still seemingly relevant, e.g. No Man's Sky, Fallout 4, Far Cry 5 and ARK: Survival World as action, RPG, open-world games. This recommendation should be good enough to recommend to user who prefer game like Assassin's Creed: Odyssey.

**Note**: I was the one who play these stuff games so I subjectively enough can say last two recommendation system indeed suggests good recommendations.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

# 2. Neighborhood Collaborative Filtering

<p align = center>
<img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*3ALliiz9hG79_2xopzgyrQ.png" width=60%>
</p>

## Utility Matrix

In recommendation system, there are interaction between two entities of users and items, users have preference for certain items, and these preference must be teased out of the data. These interaction of user-item pair is representated as a *utility matrix* which a value represents what is known about the degree of preference of user-u for item-i. This matrix is commonly sparse, because the most entries are NaN or Unknown. an Unknown value implies we have no explicit information about certain user-u preference for the item-i.

<p align=center>
<img src="https://i.stack.imgur.com/4ncGL.png" width=30%>
</p>

Utility matrix above represents information of explicit data such as given rating from an user to an item. Since utility matrix is sparse, there should be more 'NaN' or '0' values. Notice that most user-item pairs have 0 value, meaning the user has not rated the movie. The goal of most recommendation system is to predict the 'NaN' values in utility matrix.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Item-to-Item Neighbor CF
---------

<p align=center>
<img src="pics/image-6.png" width=50%>
</p>

The goal is recommend games that user might like, so we predict rating the user given for certain games, the highest rating the user given indicates the user is prefer the game.

Neighborhood Collaborative Filtering Approach work by finding similarities either in users or items and calculate predicted rating averaging rating from its neighbor. Predicted rating function is derived below with baseline function to handle such of user and item bias since user has varying rating scale, some gave higher rating others gave lower to an item while some items also are in average rated higher than others.

Lets given a two user of user-u and user-j with two item-i and item-h, for item-to-item CF, we will defined similarity between item-i and item-h.

$$\hat{r}_{ui} = baseline_{u,i} + \frac{\sum_{j∈N(u)}(Similarity(i,h). \ (r_{ji} - baseline_{u,h}))}{\sum_{j∈N(u)} Similarity(i,h)}$$

$\hat{r}_{ui}$ = Predicted ratings from user u on item i<br>
$r_{ji}$ = ratings from user j on item i<br>
$\sum_{j∈N(u)}$ = sum through all neighbors from user u<br>

$baseline_{u,i}$ = $\mu_{global} + b_{user u} + b_{item i}$

The input data is utility matrix that defined user-item interaction, could be rating given by user-u to item-i. Next computed using Neighborhood CF model, in this case is using `KNNBaseline` function from `surprise` library with number of *k : 15* and using *pearson correlation* similarity function.

```python
rating_data.pivot(index='user_id', columns='app_id', values='rating')
```

```python
from surprise import KNNBaseline

#create dictionary of parameter
params_item_item = {'k':list(np.arange(start=5, stop=40, step=5)),
          'sim_options':{'name':['cosine','pearson_baseline'],'user_based':[False]}}

# Tuning item-to-item collaborative filtering
tuning_item = RandomizedSearchCV(algo_class=KNNBaseline, param_distributions = params_item_item,
                   cv=5
                   )
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## User-to-User Neighbor CF
--------

<p align=center>
<img src="pics/image-7.png" width=50%>
</p>

In user-to-user CF, we will predict rating that will given by user to certain item based on similarity between the users. For simple example is, if user-1 is watched film A and B while user-2 is also watched film A and B but user-2 have watched film C, then film C will be recommend to user-1 since both have watched same film.

The prediction function for user-to-user is similar with the function for item-to-item CF but rather than compute similarity between item-i and item-h we will compute similarity between user-u and user-j

$$\hat{r}_{ui} = baseline_{u,i} + \frac{\sum_{j∈N(u)}(Similarity(u,j). \ (r_{ji} - baseline_{j,i}))}{\sum_{j∈N(u)} Similarity(u,j)}$$

$\hat{r}_{ui}$ = Predicted ratings from user u on item i<br>
$r_{ji}$ = ratings from user j on item i<br>
$\sum_{j∈N(u)}$ = sum through all neighbors from user u<br>

$baseline_{u,i}$ = $\mu_{global} + b_{user u} + b_{item i}$

For user-to-user, number of *k* is 5 and using *cosine similarity* function to compute similarity between users.

```python
from surprise import KNNBaseline

#create dictionary of parameter
params_user_user = {'k':list(np.arange(start=5, stop=40, step=5)),
          'sim_options':{'name':['cosine','pearson_baseline'],'user_based':[True]}}

# Tuning user-to-user collaborative filtering
tuning_user = RandomizedSearchCV(algo_class=KNNBaseline, param_distributions= params_user_user,
                                 cv=5
                                 )
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

#### Recommendation System Evaluation: RMSE

<p align=center>
<img src="pics/image-8.png" width=60%>
</p>

We got the **item-to-item CF** as the best model on predicted rating with **rmse : 0.691921**, followed by item-to-item CF with **rmse: 0.692955**. However, in this function we just calculate the predicted rating given without context-aware added into the model, so it prone to recommend the most similar item to certain user that likely come from similar game title or game developer by ignoring its released date range or price range at the first place. 

This might affect the user to reluctant to buy the same title games again (e.g Assassins Creed Franchise, Far Cry Franchise, etc.) or unaffordable game (high price games). However, some user sometimes more likely to play game from same franchise again and again, and its good to show those games in offering since it can increase likelihood of user to buy.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

# 3. Context-Aware Factorization Machines
--------

### Factorization Machines

Previously, we learn that rating prediction is generated by user, item and bias interaction. However, some user prefer to watched certain types of film or played certain genre of games. user-u is playing games item-i which is RPG and Action-shooter, hence next recommendation should be games with RPG and Action-Shooter genres. This concept is similar to applying filter in marketplace search enginee, while user want to search certain games, they will start by filter it for its genre, year release date, or anything that related to that games.

Added context such as genre, will add more dimensionality to our utility matrix for each context. So we can reshape it to become like common machine learning task data, where it have pair of <X,y> where X is predictor and y is target.

<p align=center>
<img src="pics/image-9.png" width=70%>
</p>

Since we have more feature of users, items, and genres, we can apply regression function to predict rating (y):

$$ f(x) = w_{user}.user + w_{item}.item + w_{genre}.genre $$

However, we may needed to consider user and item interaction as factorization machine approachs. The model equation for a factorization
machine of degree d = 2 is defined as (Rendle, 2010):

$$ f(x) = w_{0} +   \sum_{n=i} w_{i}.x_{i} + \sum_{n=i}\sum_{j=i+1} w_{i,j}.x_{i}.x_{j} $$

$$ f(x) = w_{0} +   \sum_{n=i} w_{i}.x_{i} + \sum_{n=i}\sum_{j=i+1} <v_{i},v_{j}>x_{i}.x_{j} $$

$w_{0}$ = intercept <br>
$w_{i}$ = feature weight or 'coefficient'<br>
$w_{i,j}$ = interaction weight or 'interaction'<br>
$v_{i},v_{j}$ = latent factor value<br>
$x_{i},x_{j}$ = feature value<br>

The first part of the FM model contains the unary interactions of each input variable $x_{i}$ with the target—exactly as in a linear regression model. The second part with the two nested sums contains all pairwise interaction of input variabls, that is $x_{i},x_{j}$.The important difference to standard polynomial regression is that the effect of the interaction is not modeled by an independent parameter $w_{i,j}$ but with a factorized parametrization $w_{ij} ≈ \ <v_{i},v_{j}> \ = \sum^{k}_{f}v_{i,f}.v_{j,f}$, which corresponds to the assumption that the effect of pairwise interactions has a low rank. This allows FM to estimates reliable parameter even in highly sparse data where standard models fail (Rendle, 2012).

$$ f(x) = w_{0} + \sum^{n}_{i=1}w_{i}.x_{i} + \frac{1}{2}\sum^{k}_{f=1}[(\sum^{n}_{i}v_{i,f}.x_{i})^2 - \sum^{n}_{i=1}v_{j,f}^2.x_{j}^2] $$

$v_{i,f}$ = latent factor value from feature i at position f (scalar) <br>
$k$ = number of latent factor

Factorization machines can estimate interactions even in high sparsity data because they break the independence of the interaction parameters by factorizing them.

The FM model of order d = 2 can be extended by factorizing ternary and higher-order variable interactions. The higher-order FM model is followed by (Rendle, 2010):

$$\hat{y(x)} := w_{0} + \sum^{n}_{i=1}w_{i}x_{i} + \sum^{d}_{l=2}\sum^{n}_{j=i} ... \sum^{n}_{j_{d}=j_{d-1}+1}(\prod^{l}_{i=1}x_{j}) \sum^{k}_{f=1}\prod^{l}_{i=1}v_{j,f} $$

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### FM: Regression Optimization Tasks

The objective of this task is to minimize squared error of predicted value - true value, and loss function that commonly use is Mean Squared Error (MSE).

$$Objective = min(\frac{1}{2}\sum^{N}_{i=1}(y_{i} - \hat{y_{i}}^2)) $$

Since out feature is highly sparse, prediction model might lead to overfitting. To avoid overfitting case, Regularization term such as L2 Norm (Ridge Regularization) can be added to cost function. Three optimzation methods have been proposed for FM: stochastic gradient descent
(SGD) [Rendle 2010], alternating least-squares (ALS) [Rendle et al. 2011], and Markov Chain Monte Carlo (MCMC) inference [Freudenthaler et al. 2011].

$$Objective = min\frac{1}{2}\sum^{N}_{i=1}[(y_{i} - \hat{y_{i}}^2)] + \frac{1}{2}\sum_{i=1}[\alpha{}(w_{i}^2 + \lVert v_{i} \rVert^2)]$$

### FM: Regression Model

In this section, dataset from Table 3 is used. FM model model takes sparse matrices of `scipy.sparse` as its feature input. [`DictVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html) from `sklearn.feature_extraction` is applied to features (X) to transforms the categorical variables (user_id, app_id, price_rate) into a one-hot encoded vectors. Table 3 in our project will be used on this model.

```python
from sklearn.feature_extraction import DictVectorizer

v = DictVectorizer()

X_fm = X_train.to_dict(orient='records')
y_fm = np.asarray(y_train.values)

X_fm = v.fit_transform(X_fm)
```

[`FMRegressor`](https://github.com/tohtsky/myFM) from `myFM` libraries is used to predict rating of context-aware factorization machines. This model apply **Bayesian Factorization Machines** with **Gibbs sampling** or **Markov Chain Monte Carlo (MCMC)** algorithm to optimizing the parameter model rather than **Stochastic Gradient Descent**. parameter *rank (d)* is hyperparameter that is used in here to defined factorization machine of degree that will be used as the model.

```python
fm_regression = MyFMRegressor(rank=2)

fm_regression.fit(X_fm, y_fm)

y_pred_fm = fm_regression.predict(X_test)
```

Let's say if we do not only want interaction term regarding price range of games but also its release date range. Hence we can simply attached new features of release date but try to increase parameter factorization machine of degree to obtain better prediction toward high dimensionality data.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### FM Regression Recommendation 

<p align=center>
<img src="pics/image-10.png" width=70%>
</p>

It's show up that **Context-Aware Factorization Machine** with **Monte Carlo Markov Chain (MCMC)** Optimization is indeed having better *RMSE* score in **0.1461** rather than **Stochastic Gradient Descent** with *4.652*. This model is also give less *RMSE* score compare to previous Neighborhood Collaborative Filtering. By adding more context like released date feature into model, it increase model performance by reducing *RMSE* into **0.1368**. However adding released date context, it increase factorization machine of degree into 8 hence require more computational cost.

Let's try this recommendation to get N-recommendation games on certain user IDs. Below is few games that has played by user **3924036**

<p align=center>
<img src="pics/image-11.png" width=70%>
</p>

Here are the recommendation of of two price range based on user IDs **3924036** preference, the **first 10<sup>th</sup>** are the recommendation games for **medium price range** while the **last 10<sup>th</sup>** are the recommendation games for **expensive price range**. According to our objective to refer medium price range for user IDs to increase likelihood of the user to buy the games, the medium price range recommendation should be good for the user. However, the recommendation on expensive range is seemingly more relatable to history log of user IDs **3924036**. The recommendation should be followed the history tracking of the user capability to buy the games, in this example User IDs **3924036** seem has capability to afford expensive or even luxury price games and prefer free games rather than the other cheap games, hence the recommendation is become more randomize.

<p align=center>
<img src="pics/image-12.png" width=70%>
</p>

### FM: Classification Objective

Classification problem calculate probability that x is belong to class 1, so sigmoid function is used to approximate the probability of the function.

$$\sigma{(\hat{y}(x))} = p(x) = \frac{1}{1 + e^{-(\hat{y}(x))}} $$

The objective of classification task is to maximize the probability of a class.

$$Objective = \max_{w}[\prod^{N}_{i=1}P(y_{i}|X^{(i)},w)] $$

Multiplication function will cost computational time greatly, then we work on a logarithmic scale, because PDF term is additive.

$$Objective = \min_{w*,v*}[-\frac{1}{N}\sum_{iED}(y^{(i)}.log(p(x^{(i)})) + (1-y^{(i)}).(1-log(p(x^{(i)}))))] + [\frac{1}{2}\sum_{m}(w^2 + \lVert v_{m} \rVert^2)] $$

- $y_{i}$ = True Class
- $p(x^{(i)})$ = Probability that x belongs to class 1
- w* = weight from feature
- v* = latent factor for feature

### FM: Classification Models

In this section, dataset from Table 4 is used. FM model model takes sparse matrices of `scipy.sparse` as its feature input. [`DictVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html) from `sklearn.feature_extraction` is applied to features (X) to transforms the categorical variables (user_id, app_id, price_rate) into a one-hot encoded vectors. Table 3 in our project will be used on this model.

```python
from sklearn.feature_extraction import DictVectorizer

v = DictVectorizer()

X_fm = X_train.to_dict(orient='records')
y_fm = np.asarray(y_train.values)

X_fm = v.fit_transform(X_fm)
```

[`FMClassification`](https://github.com/tohtsky/myFM) from `myFM` libraries is used to predict class probability of context-aware factorization machines. This model apply **Bayesian Factorization Machines** with **Gibbs sampling** or **Markov Chain Monte Carlo (MCMC)** algorithm to optimizing the parameter model rather than **Stochastic Gradient Descent**. parameter *rank (d)* is hyperparameter that is used in here to defined factorization machine of degree that will be used as the model.

```python
fm_classification = myFMClassification(rank=2)

fm_classification.fit(X_fm, y_fm)

y_pred_fm = fm_classification.predict(X_test)
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### FM: Classification Models Performance

<p align=center>
<img src="pics/image-13.png" width=50%>
</p>

Accuracy score given by Classification model to predict whether certain User IDs would recommend or not a game IDs is 0.92. This score is in good range scale to consider in 1000_000 instances data prediction. However both rating prediction or class of user to recommend the game or not is defined by explicit data that commonly given by many users.

# Alternating Least Square
-----

Bayesian Personalized Ranking algorithm return implicit value of user-item pair into pairwise ranking. Pairwise ranking simply return consumed item as positive item while unconsumed value as Negative item, hence Positive item > Negative item. Based on those concept the algorithm use to update parameter to maximize probability that user like positive item over negative item. We observe that previously model predict each score to item individually, hence our parameter is optimized individually. The issue is we have parameter that focus on giving correct prediction only, not correct ordering. We can compare positive item (consumed item) is more preferred by user over negative item (unconsumed item).

Our training data consists of User IDs, Positive sample items (+) and Negative sample items (-). In matrix factorization, utility value such as rating is predicted using $r_{ui} = x_{u}.y^{T}_{i} $. In BPR, we can have confidence term by comparing predicted utility of $\hat{r_{ui}}$ and $\hat{r_{uj}}$ with $ \hat{r_{uij}} = \hat{r_{ui}} - \hat{r_{uj}} $. Sigmoid function is applied to maximize the probability that $r_{ui}$ is bigger than $r_{uj}$.

$$ \hat{r}_{uij} = \hat{r}_{ui} - \hat{r}_{uj} $$

$$ \sigma{(\hat{r}_{uij})} = \frac{1}{1+e^{-\hat{r_{uij}}}} $$

Using Bayes Theorem, we want to finding parameter of likelihood given predicting user u preferred item i > j

$$ P(\theta|i>_{u}j) = \frac{P(i>_{u}j|\theta).P(\theta)}{P(i>_{u}j)} $$

$P(\theta|i>_{u}j)$ = Probability of parameter $\theta$ given $i>_{u}j$ (posterior probability/training data)
$P(i>_{u}j|\theta)$ = Probability of $i>_{u}j$ given parameter $\theta$ (likelihood/target)
$P(\theta)$ = Probability of parameter $\theta$ (prior)
$P(i>_{u}j)$ = Probability of $i>_{u}j$ (marginal probability/training data)

Since $P(i>_{u}j)$ is constant (observed from training data), we can reduce by

$$P(\theta|i>_{u}j) = P(i>_{u}j|\theta).P(\theta) $$

$$P(\theta|i>_{u}j) = \sigma{(\hat{r}_{ui} - \hat{r}_{uj}).P(\theta)} $$

Our goal to find optimal parameter given our training data, what we can do during training process with all our training samples:

$$Objective = \max \prod_{u,i,j}\sigma{(\hat{r}_{ui} - \hat{r}_{uj}).P(\theta)}$$

Since multiplication is expensive on computational time, we could multiply by log natural or logarithmic form to make computational become addition

$$ Objective = max \sum_{u,i,j}ln(\sigma{(\hat{r}_{ui} - \hat{r}_{uj})}).ln(P(\theta)) $$

Using uniform prior concept with mean=0 and standard deviation=1 we can solve $ln(P(\theta))$ into $f(x) = -\frac{\lVert x_{u} \rVert^2}{2} $ hence our function is consist of User factor $(x_{u})$, item i factor $(y_{i})$ and item j factor $(y_{j})$. 

With $P(i>_{u}j|\theta) = \sigma{(\hat{r}_{ui} - \hat{r}_{uj})}$, and applying logarithmic scale to change multiplication to addition, we can replace the function into:

$$Objective = \max_{x*,y*}\sum_{u,i,j} ln(\sigma{(x_{u}.y^{T}_{i} - x_{u}.y^{T}_{j})}) + ln(P(\theta))$$

By adding L2 Regularization to overcome overfitting model and turn the model into Negative Log Likelihood to optimize the model parameter, the function we will use is

$$Objective = \min_{x*,y*}\sum_{u,i,j} - ln(\sigma{(x_{u}.y^{T}_{i} + x_{u}.y^{T}_{j})}) + \frac{\lambda \lVert x_{u} \rVert^2}{2} + \frac{\lambda \lVert y_{i} \rVert^2}{2} + \frac{\lambda \lVert y_{j} \rVert^2}{2} $$

### BPR Models

Bayesian Personalized Ranking used in this project is using **`bpr`** from [**`implicit`**](https://github.com/benfred/implicit) libraries toward dataset of consist user IDs, app IDs, and hours user spent on playing certain games, turn into sparse coo matrix in aim to reduce computational cost since we have sparse matrix.

```python
def preprocess_implicit_data(data,feature):
    row = data.user_id.values
    col = data.app_id.values
    value = data[feature].values

    utility_matrix = sp.coo_matrix((value, (row, col)))

    return utility_matrix

>>> <19886x2948 sparse matrix of type '<class 'numpy.float64'>'
	with 20000 stored elements in COOrdinate format>

from implicit.evaluation import train_test_split

# split data into train and test
train_full,test = train_test_split(implicit_hour,train_percentage=0.8)

import optuna

# import model alternating least squares
from implicit import als
from implicit.evaluation import ndcg_at_k

def als_tuning(trial):
    factors = trial.suggest_int(name='factors',
                                             low=10,
                                             high=100,step=10)
    regularization = trial.suggest_float(name='regularization',
                                         low=0.0001,
                                         high=1.0,
                                         step= 0.001)

    alpha = trial.suggest_float(name='alpha',
                                      low=0.001,
                                      high=1.0,
                                      step= 0.001)

    # instanciate model
    model_als = als.AlternatingLeastSquares(factors=factors,
                                            regularization= regularization,
                                            alpha= alpha)

    # fit model
    model_als.fit(train)

    # test
    val_metrics = ndcg_at_k(model = model_als, train_user_items= train,
                                  test_user_items = val,K=5)
    return -val_metrics

# minimizing is equal to
study_als = optuna.create_study(direction="minimize")
study_als.optimize(als_tuning, n_trials=2)
```

<p align=center>
<img src="pics/image-18.png" width=70%>
</p>

Previously we are already used Neighborhood, Factorization Machines to predict rating and return top-N recommendation games based on User IDs **3924036**, now we are using **Alternating Least Square** to return top-N of recommendation games based on User ID's activity log of Implicit data such as hours spent on few games.

<p align=center>
<img src="pics/image-16.png" width=70%>
</p>

Recommendation given by models represents few well-known games and seemingly relatable to historical games played by the User IDs **3924036**. Since the evaluation score used on rating prediction in Factorization Machine and ordered ranking games recommendation by impliciy data, the comparation between both method is difficult to tell, but since the games played by User IDs is well known to, we can compare by recommendation return based on the games genre or popular tags.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

# Conclusion
-----

- **Content-based recommendation system** based on **similarity** between the items show good suggestion but still typically give similar title or game franchise after all. However this recommendation is good for user who prefer to play same franchise games. In this example, we want to return recommendation for Assassin's Creed Odyssey games, it mostly recommend another Assassin's Creed Games.

- **Content-based recommendation with Novelty** indeed give more variability games rather than based on similarity only. The recommendation games seemingly give same genre or popular tags. But still novelty is calculated according to number of interaction created by user toward certain games that really essentials on new released games, since those games don't have big number of interactions at first place.

- **Content-based recommendation with Serependity** return more randomized games but still keep intact with similarity of searched games regarding genre or popular tags. But somehow some recommendation games found out to be irrelevant games with our searched games (Assassin's Creed Odyssey) that most likely RPG, Open-world, Action but instead give Strategy-based games like Heart of Iron IV or Recruit, despite of those game is medieval theme like Assassin's Creed Odyssey. This circumstance can be considered to reduced some popular tags of games and only find tags that really related to games. However this matter can be resolved by context-aware recommendation system whether it pre-filtered, post-filtered, or calculate interaction terms.

- Content-based is indeed a good recommendation system so far, but still few user does have their own preference to play or buy certain games. Collaborative Filtering utilize this user's preference to return more better recommendation based on another user similarity. **Neighborhood Collaborative Filtering** utilize the user preference based on neighboor users that has same similarity. The **`KNNBaseline`** from **`surprise`** used for either user-to-user or item-to-item CF to predict rating that would likely given by user to certain item. It turns out that evaluation on loss function using **RMSE having 0.691 on item-to-item and 0.692 on user-to-user collaborative filtering**. Even this recommendation turn out to be having RMSE performance but do not maximize utilization of another context as our first objective such as game's price or game's release date.

- Our objective is to recommend a games that having low-medium price for certain user to increase likelihood of user to buy the games. Hence, context like price range could be considered to predict rating of user-u to item-i with price range-c. **Context-Aware Bayesian Factorization Machines** is applied to resolved this matter. Context-Aware Factorization Machine utilize these categorical features on its interaction term toward the users-items and optimizing using **Monte Carlo Markov Chain (MCMC)** algorithm. These algorithm can also applied with two or more categorical features (e.g genre, price-range, released date-range) with high dimensionality by utilize it's factorization machine of degree. **Total RMSE evaluation implies that this recommendation system performance indeed have lower RMSE rather than Neighborhood CF**.

- This Context-Aware Bayesian Factorization Machines is indeed applicable to another context like genre-related feature when we want to create recommendation system model that more related to user's genre or even location preference.

- **Alternating Least Square** used to utilize *implicit data* like total hours user spent on few games, but ALS do have **low NDCG score** due to much unique games in dataset. However the implicit data should be utilized more with context-aware to maximize the output of recommendation system since most of the user prefer not to give rating on games they interacted with.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

# REFERENCES













