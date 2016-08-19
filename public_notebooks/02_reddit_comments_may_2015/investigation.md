
## Investigation of [May 2015 Reddit Comments](https://www.kaggle.com/reddit/reddit-comments-may-2015)

The database has one table, May2015, with the following [fields](https://github.com/reddit/reddit/wiki/JSON):
 `created_utc, ups, subreddit_id, link_id, name, score_hidden, author_flair_css_class, author_flair_text, subreddit, id, removal_reason, gilded, downs, archived, author, score, retrieved_on, body, distinguished, edited, controversiality, parent_id`
 
 --------
 
 It takes:
 - 1.2 seconds to grab 0.1 mln
 - 10 secs to grab 1 mln
 - 5 minutes secs to grab 10 mln.

Data have 54504410 comments (~55 mln), so it is expensive to get all of it with all the columns. After some preliminary analysis, I found that many columns are useless.

 - name (just an id with some prefix)
 - id (an string, which increases: cqug90g, cqug90h, ...)
 - author_flair_css_class, author_flair_text (text and css class of authors flair
 - downs, archived (is always 0)
 - retrieved_on (when was the comment extracted by API parser)
 - score (the same value as ups)
 - subreddit_id (is the ID of subreddit)
 - score_hidden (whether the comment score is currently hidden)
 - removal_reason (s always None)
 
A few other fields that were not helpful for me: `link_id`, `parent_id`, `edited`, `distinguished`. 

The last one takes only 4 different values and if I ever will need to use it here is a snippet:

    type_distinguished = {'moderator': 1, 'admin': 2, 'special': 3}
    df['distinguished'] = df['distinguished'].apply(lambda x: type_distinguished.get(x, 0))
    
The most interesting data will be most probably in the body statement. I will leave this investigation to another notebook. Will take a look at the low-hanging fruits and prepare a list of interesting questions to ask during the NLP part.


```python
import pandas as pd
import numpy as np
import sqlite3

import matplotlib.pyplot as plt
import seaborn as sns 
%matplotlib inline
```


```python
%%time
query = """
    SELECT created_utc, ups, subreddit, gilded, author, controversiality
    FROM May2015
"""
df = pd.read_sql(query, sqlite3.connect('input/database.sqlite'), parse_dates=['created_utc'])
```

    CPU times: user 6min 15s, sys: 6min 31s, total: 12min 47s
    Wall time: 22min 1s


The original file is 30Gb, and even after removing some columns it is still big. After investigating the values, I found that `controversiality` is boolean. Also found that the maximum value of `gilded` is < 256 and the maximum value of `ups` is < 32k.

So I am just trying to decrease the amount of memory needed.


```python
%%time
df['controversiality'] = df['controversiality'].apply(lambda x: x == 1)
df['gilded'] = df['gilded'].astype(np.int8)
df['ups'] = df['ups'].astype(np.int16)
```

    CPU times: user 16.2 s, sys: 5.01 s, total: 21.2 s
    Wall time: 32.9 s


### Time distribution
Let's see how does the distribution between number of comments created each day looks like.


```python
%%time
fig=plt.figure(figsize=(20, 10))

df.groupby(df['created_utc'].map(lambda x: x.weekday())).size().plot(
    'bar', title='Number of comments at each day of a week', ax=fig.add_axes((0, 0, 0.5, 0.5))
);
df.groupby(df['created_utc'].map(lambda x: x.hour)).size().plot(
    'bar', title='Number of comments at each hour', ax=fig.add_axes((0.53, 0, 0.5, 0.5))
);
df.groupby(df['created_utc'].map(lambda x: x.day)).size().plot(
    'bar', title='Number of comments at each day', ax=fig.add_axes((0, -0.55, 1.03 ,0.5))
);
```

    CPU times: user 11min 12s, sys: 7min 37s, total: 18min 50s
    Wall time: 22min 10s



![png](output_6_1.png)


As we see Friday has the biggest amount of comments (a day before weekends, people have nothing to do).

When I looked at the distribution for each hour, I saw an interesting trend. The bars look like a sine wave with the lowest ~9 and the highest at 19. This would not be strange if the timestamp would be in the timezone of the person who commented (because clearly there are more people who follow a regular active/sleep cycle). But here we are speaking about international audience which means that they come from different timezones. **So may be the audience is not so international as I expected.**

The last graph shows that this trend continues during the longer period of time, and that the number of comments at each specific day of the week is not really changing.

When I have done the distribution between the minutes, the bars looked approximately the same. This means that there is no real difference between the number of comments sent at different minutes.

---- 
### Subreddits

Let's investigate the comments in the various subreddits. It is clear that there will be a few very very popular subreddits with huge amount of comments and many many subreddits with close to 1 comment. It would be also interesting to take a look at the most popular subreddits. 

Before I will move ahead, I will create a helper function that creates 2 plots (I will be using it often).


```python
def plot_distribution_popular(df, title1, title2, num=60):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))
    
    df.plot(title=title1, xticks=[], logy=True, ax=axes[0]);
    df[:num].plot(title=title2, kind='bar', logy=True, ax=axes[1]);
```


```python
%%time
plot_distribution_popular(
    df.groupby(df['subreddit']).size().sort_values(ascending=False),
    "Distribution of number of comments in each subreddit",
    "Number of comments for the most 60 most popular subreddits"
)
```

    CPU times: user 23.1 s, sys: 26.1 s, total: 49.2 s
    Wall time: 1min 45s



![png](output_9_1.png)


As we see from the first plot $~1/5$ of all subreddits have only 1 comment, ~$1/2$ has less than 10 comments. Nothing really surprising about most popular subreddits: people like movies, music, games, random blabbing.

**Interesting question to NLP:**
> can we distinquesh between very popular subreddits based on the text body

### Authors
Now let's see who writes these comments. Expect to see the same long tail stuff. 


```python
%%time
plt.figure(figsize=(20, 4))
df.groupby(df['author']).size().sort_values(ascending=False)[:60].plot(
    'bar', logy=True, title="Number of comments for the 60 most popular authors"
);
```

    CPU times: user 2min 4s, sys: 8min 20s, total: 10min 25s
    Wall time: 21min 9s



![png](output_11_1.png)


It looks like there are a lot of comments which are most probably deleted. Also some of them automoderated. It makes sense to take a look at them.

Looking at `df[df['author'] == 'AutoModerator'].value_counts().sort_values(ascending=False)` I saw that all the body text is huge wall of text of of automatic warning that just tells something like this 'you have done something wrong, so we removed it'. 

Similar thing is with `df[df['author'] == 'AutoModerator'].value_counts().sort_values(ascending=False)`. Huge majority (99%) is just value `[deleted]`. Others are either an automated text or a short ok/thank/you too messages. 

Do not see any value having this around, so I will just get rid of them


```python
%%time
df = df[(df['author'] != '[deleted]') & (df['author'] != 'AutoModerator')]
```

    CPU times: user 32.2 s, sys: 2min 4s, total: 2min 36s
    Wall time: 5min 40s



```python
%%time
plot_distribution_popular(
    df.groupby(df['author']).size().sort_values(ascending=False),
    "Distribution of number of comments for each author",
    "Number of comments for the 60 most popular authors",
)
```

    CPU times: user 2min 37s, sys: 8min 28s, total: 11min 5s
    Wall time: 21min 40s



![png](output_14_1.png)


From the first plot we see that a hope for humanity was partially restored. $~1/3$ of the authors posted 1 comment and $~3/4$ posted less than 10 comments.

Apparently some people have absolutely nothing to do with their life. $10^4$ comments on reddit in a month. This is 14 comments per hour non stop. Some of them suggest that they are bots (`autowikibot`, `politicBot`, but I believe that all top 60 are bots.

Would be interesting to investigate their speech pattern, where do they post and how many upvotes do they have.

**Interesting question to NLP:**
> can we distinquesh between very popular authors based on the text body


----
### Subreddits and authors

Now let's combine both subreddits and authors and find subreddits with the biggest amount of authors.


```python
%%time
plot_distribution_popular(
    df.groupby('subreddit')['author'].nunique().sort_values(ascending=False),
    "Distribution of author in subreddits",
    "Number of authors in the 60 most popular subredddits",
)
```

    CPU times: user 12min 4s, sys: 1h 15min 34s, total: 1h 27min 39s
    Wall time: 3h 39min 58s



![png](output_16_1.png)


----
### Scores
Let's see how the comments were recieved by the audience. 

I would expect that a lot of comments get close to 0 score, I also would expect votes to be skewed towards positive values and clearly there will be a long tail. Investigating `df['ups'].value_counts()` shows that my expectations were wrong. **Majority of the comments have 1 vote.** Then 2, 3, and only then 0 votes.

This clearly need more detailed investigation, because the grouping is not smart (I have to group comments). I will not do anything with near 0 votes, but I will group highly positive and highly negative comments.


```python
print len(df[df['ups'] < -10]), len(df[df['ups'] < -50]), len(df[df['ups'] < -100]), len(df[df['ups'] < -500])
print len(df[df['ups'] > 10]), len(df[df['ups'] > 50]), len(df[df['ups'] > 100]), len(df[df['ups'] > 500]), len(df[df['ups'] > 1000]), len(df[df['ups'] > 5000])
```

    243602 12819 2487 23
    3977268 750247 352454 55865 22687 136



```python
down_ranges = [-5000, -1000, -500, -100, -50, -10]
up_ranges   = [10, 50, 100, 500, 1000, 5000, 10000]

for i in xrange(len(down_ranges) - 1):
    s, e = down_ranges[i], down_ranges[i + 1]
    df.loc[(df['ups'] >= s) & (df['ups'] < e), 'ups'] = (s + e) / 2
    
for i in xrange(len(up_ranges) - 1):
    s, e = up_ranges[i], up_ranges[i + 1]
    df.loc[(df['ups'] > s) & (df['ups'] <= e), 'ups'] = (s + e) / 2
```


```python
plt.figure(figsize=(20, 4))
df['ups'].value_counts().plot('bar', logy=True, title="Distribution of votes. Big votes is the aggregation of votes at that region");
```


![png](output_20_0.png)


Majority of the comments recived a positive feedback. There is higher probability that a comment will recive from 10 till 50 upvotes than it will recive 0 upvotes. Look at the position of the first negative score (10-th from the left). And just for fun, the last look at the distribution of the positive/negative/neutral comments.


```python
pd.Series({
    'positive': len(df[df['ups'] > 0]),
    'neutral': len(df[df['ups'] == 0]),
    'negative': len(df[df['ups'] < 0])
}).plot('bar', logy=True);
```


![png](output_22_0.png)


**Interesting question to NLP:**
> What makes a comment highly positively/negatively pursuited by the crowd. 

### Gilded comments
As far as I understood, gilded comment means that some other user endorsed the comment (by making some monetary donation). It would be interesting to see the score distribution for these comments.


```python
plt.figure(figsize=(20, 4))
df[df['gilded'] > 0]['ups'].value_counts().plot('bar', logy=True);
```


![png](output_25_0.png)


As we see clearly gilded comments have reasonably high score. But we also see that sometimes the comment is gilded nonetheless of it's unpopularity (~-30 or even some with ~-300)

### Controversiality
Previously I converted `controversiality` to a boolean value. Now it is time to ivestigate it. Based on my understanding, this field is marked when during some time a comment received significant amount of upvotes and downvotes.

Let's check for the most controversial authors, the most controversial subreddits and the scores of controversial comments.


```python
print "Percent of controversial comments", len(df[df['controversiality']]) / float(len(df))
```

    Percent of controversial comments 0.0240061201902



```python
plt.figure(figsize=(20, 4))
df[df['controversiality']]['ups'].value_counts().plot('bar', logy=True);
```


![png](output_28_0.png)


The distribution of scores in controversial comments is significantly different from the distribution of all comments and the distribution of gilded comments. They are dominated by small score with the maximum at 0.

Ok, with scores everything was simple, but in order to find controversial authors or controversial subreddits (the approach will be the same) it is not enough just to group by a topic and sort. This is because clearly a topic with 10k comments has higher chance of getting 10 controversial comments than a topic with 15 comments.

To overcome this I will caclulate the percent of controversial comments in each topic and then exclued topics with less than min_size comments.


```python
def analyzeControversial(column, min_size):
    res = df[[column, 'controversiality']].groupby(column).mean()
    tmp = res.join(pd.DataFrame(df.groupby(df[column]).size(), columns=["cnt"]))
    return tmp[tmp['cnt'] >= min_size].sort_values('controversiality', ascending=False)    
```


```python
%%time
tmp = analyzeControversial('subreddit', 1000)
```

    CPU times: user 43.6 s, sys: 1min 28s, total: 2min 11s
    Wall time: 4min 28s



```python
tmp.head(n=20)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>controversiality</th>
      <th>cnt</th>
    </tr>
    <tr>
      <th>subreddit</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>UkrainianConflict</th>
      <td>0.192875</td>
      <td>17908</td>
    </tr>
    <tr>
      <th>samharris</th>
      <td>0.171821</td>
      <td>1455</td>
    </tr>
    <tr>
      <th>russia</th>
      <td>0.161770</td>
      <td>13581</td>
    </tr>
    <tr>
      <th>nl_Kripparrian</th>
      <td>0.141176</td>
      <td>1190</td>
    </tr>
    <tr>
      <th>svenskpolitik</th>
      <td>0.137472</td>
      <td>3157</td>
    </tr>
    <tr>
      <th>Polska</th>
      <td>0.128440</td>
      <td>1744</td>
    </tr>
    <tr>
      <th>serialpodcast</th>
      <td>0.126156</td>
      <td>40212</td>
    </tr>
    <tr>
      <th>Israel</th>
      <td>0.112905</td>
      <td>10965</td>
    </tr>
    <tr>
      <th>Conservative</th>
      <td>0.108730</td>
      <td>11947</td>
    </tr>
    <tr>
      <th>Trans_fags</th>
      <td>0.107634</td>
      <td>2109</td>
    </tr>
    <tr>
      <th>baltimore</th>
      <td>0.106740</td>
      <td>15074</td>
    </tr>
    <tr>
      <th>olympia</th>
      <td>0.105111</td>
      <td>1037</td>
    </tr>
    <tr>
      <th>QuotesPorn</th>
      <td>0.104779</td>
      <td>4896</td>
    </tr>
    <tr>
      <th>TACSdiscussion</th>
      <td>0.103864</td>
      <td>8463</td>
    </tr>
    <tr>
      <th>syriancivilwar</th>
      <td>0.098112</td>
      <td>41677</td>
    </tr>
    <tr>
      <th>sanfrancisco</th>
      <td>0.096774</td>
      <td>10292</td>
    </tr>
    <tr>
      <th>kurdistan</th>
      <td>0.096429</td>
      <td>1120</td>
    </tr>
    <tr>
      <th>LegionOfSkanks</th>
      <td>0.093750</td>
      <td>1024</td>
    </tr>
    <tr>
      <th>blog</th>
      <td>0.092480</td>
      <td>13192</td>
    </tr>
    <tr>
      <th>SRSDiscussion</th>
      <td>0.092072</td>
      <td>1564</td>
    </tr>
  </tbody>
</table>
</div>



As we see, majority of controversial comments come from controversial topics (politics), but it looks like podcasters and masturbators also have things they can't agree upon.

Lets look at the least controversial stuff.


```python
print [str(i) for i in tmp[tmp['controversiality'] == 0].index.values.tolist()]
```

    ['superleague', 'inFAMOUSRP', 'Thaumaturgy', 'TeamRedditTeams', 'hubchargen', 'huntersbell', 'TapTapInfinity', 'identifythisfont', 'TTPloreplaycentral', 'sufficiencybot', 'indiegameswap', 'infertility', 'SupersRP', 'Stuff', 'SteamPunkPowers', 'kancolle_ja', 'TheInnBetween', 'TheRedLion', 'watchinganime', 'AsTheClockTurns', 'TheVeneration', 'ArtJunkie', 'hcteams', 'havoc_bot', 'TotalDramaRoleplay', 'TotalDramaRoleplay2', 'TotalDramaWerewolf', 'AquamarinesDen', 'hampan', 'AppNana', 'TradeOrGift', 'Triumph', 'TroveCreations', 'kinksters_gone_wild', 'ShadowBan', 'SburbRP', 'maddenmobilebuysell', 'mcservers', 'PowerShell', 'PrivateFiction', 'BeautyAddiction', 'RPGStuck', 'BeardedDragons', 'BayStars', 'RandomActsOfTf2', 'RandomActsofCards', 'BasketballGMFantasy', 'RandomActsofMakeup', 'BankBallExchange', 'usedpanties', 'BakaNewsJP', 'uvtrade', 'SaintsFC', 'BTFC', 'lowlevelaware', 'BBWGW', 'RapWars', 'RateMyMayor', 'RecruitCS', 'vapeitforward', 'AutoModerator', 'Riftvielrpg', 'RunnerHub', 'legotrade', 'learnjavascript', 'Askasurvivor', 'SVExchange', 'TrueSTL', 'UHCMatches', 'Unity2D', 'cancer', 'bangalore', 'wrestlingisreddit', 'baseballcirclejerk', 'wsgy', 'earlyPowers', 'dragonblaze', 'xTrill', 'doommetal', 'ALORP', 'braswap', 'discexchange', 'ACON_Support', 'digimonrp', 'buildmeapc', 'cardfightvanguard', 'wowguilds', 'dailyprogrammer', '90daysgoal', 'csshelp', '6thForm', '50B', '4Runner', 'clothdiaps', 'csgoscores', 'youtubers', 'cock', '3dsFCswap', 'cryptospread', 'cryptoparadise', 'compDota2', 'espnyankees', 'animelegwear', 'UsenetInvites', 'freedesign', 'gonewildaudio', 'VerdantFantasyRP', 'wheelanddeal', 'VictorianWorldPowers', 'giftcardexchange', 'Vivillon', 'AnimeSketch', 'WeAllGoWild', 'gamemaker', 'WonderTrade', 'XMenRP', 'fringefashion', 'friendsafari', 'freedonuts', 'YAwriters', 'airsoftmarket', 'fragsplits', 'YGOBinders', 'YGOSales', 'YamakuHighSchool', 'YasoHigh', 'YouEnterADungeon', 'woiafpowers', 'fnafcringe', 'Yugioh101', 'ZMR', 'ableton', 'AdoptMyVillager', 'fivenightsatfangames', 'fireemblemcasual', 'PotterPlayRP', 'PostWorldPowers', 'Pokemongiveaway', 'realtech', 'CookieCollector', 'GalacticGuardians', 'GameTrade', 'GameofThronesRP', 'rwbyRP', 'GetFairShare', 'telescopes', 'GlobalPowers', 'rotmgtradingpost', 'roleplayponies', 'GoogleCardboard', 'retrogameswap', 'HGTV_Verse', 'test', 'CoCBot', 'GTAV_Cruises', 'razer', 'HazardOps', 'randomsuperpowers', 'HeistTeams', 'randomactsofdota2', 'theColdWarGame', 'randomactsofcsgo', 'rakugakicho', 'CivcraftExchange', 'HelpMeFind', 'quilting', 'HistoricalWorldPowers', 'themoddingofisaac', 'themountaingoats', 'GWABackstage', 'GIRLSundPANZER', 'protectoreddit', 'FFXIVRECRUITMENT', 'subredditreports', 'DestinySherpa', 'survivorcirclejerk', 'DegradingHoles', 'DinosaurDrawings', 'stobuilds', 'DanceDanceRevolution', 'DisneyPinTrading', 'Drag', 'DuelingCorner', 'spheremasterrace', 'E30', 'Emerald_Council', 'ExploreFiction', 'slashdiabloevents', 'GCXRep', 'skyrimrequiem', 'FIFACoins', 'FairyTailRP', 'simplerockets', 'FemBoys', 'Fibromyalgia', 'FierceFlow', 'shittyaskreddit', 'CrossStitch', 'shadownet', 'sgsflair', 'FocusST', 'Freeclams', 'selfharm', 'HogwartsRP', 'HomeworkHelp', 'Pokemonexchange', 'OzoneOfftopic', 'Motorsports_ja', 'NSFWskype', 'Needafriend', 'NewMarvelRp', 'BreedingDittos', 'newsokuvip', 'OCD', 'Omnipotent_League', 'Braveryjerk', 'OmniversePenitentiary', 'trollabot', 'myNBA2KMobile', 'Bowling', 'mueflair', 'msp', 'MonsterStrike', 'ttcafterloss', 'PJRP_Community', 'PKMNRedditLeague', 'PercyJacksonRP', 'PhascinatingPhysics', 'PixelDungeon', 'microsoftsoftwareswap', 'PloungeMafia', 'mhguildquests', 'BitTippers', 'metaljerk', 'PokemonForAll', 'uhccourtroom', 'PokemonPlaza', 'notebooks', 'MonarchyOfEquestria', 'HotPeppers', 'CampHalfBloodRP', 'CautiousBB', 'IAmAFiction', 'CasualPokemonTrades', 'ImagesOfEarth', 'ImaginaryTurtleWorlds', 'IronThroneRP', 'politicalpartypowers', 'pokemontrades', 'pkmntcgtrades', 'pillowtalkaudio', 'KitSwap', 'LabourUK', 'LeagueConnect', 'Leathercraft', 'LetsChat', 'ModerationLog', 'tomhiddleston', 'LineRangers', 'CadenMoranDiary', 'MCSRep', 'MLBStreams', 'Magicdeckbuilding', 'Magnolia_Town', 'Malazan', 'CHERUB_RP', 'MerylRearSolid', 'MicrosoftBand', 'MigrantFleet', 'translator', 'Miniswap', 'zookeeperbattle']


Wow, there is a lot of topics with significant amount of comments where non of them was marked as controversial.


```python
%%time
tmp = analyzeControversial('author', 1000)
```

    CPU times: user 3min 24s, sys: 12min 34s, total: 15min 59s
    Wall time: 25min 3s



```python
plt.figure(figsize=(20, 4))
tmp['controversiality'].plot(xticks=[], title="Controversiality of very popular authors");
```


![png](output_37_0.png)



```python
tmp.head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>controversiality</th>
      <th>cnt</th>
    </tr>
    <tr>
      <th>author</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>summer_dreams</th>
      <td>0.311516</td>
      <td>1207</td>
    </tr>
    <tr>
      <th>orion4321</th>
      <td>0.290654</td>
      <td>1070</td>
    </tr>
    <tr>
      <th>Lonly-jap</th>
      <td>0.273373</td>
      <td>1690</td>
    </tr>
    <tr>
      <th>SavannaJeff</th>
      <td>0.261029</td>
      <td>1088</td>
    </tr>
    <tr>
      <th>Tenaciousceeee</th>
      <td>0.224701</td>
      <td>1255</td>
    </tr>
  </tbody>
</table>
</div>



As we see there is a small amount of users who constantly post controversial comments (may be they are trolls). Amost all popular commenters have at least one comment marked as controversial. And approximately 1/5 of popular commenters have never posted anything controversial.

This is all for todays analysis, will just remind a few topics for NLP:

 - can we distinquesh between very popular subreddits based on the text body
 - can we distinquesh between very popular authors based on the text body
 - what makes a comment highly positively/negatively pursuited by the crowd
 - what makes a comment controversial


```python

```
