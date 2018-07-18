# Named-Entity-Recognition

We have created a dataset of Hindi-English Code-Mixed Social Media Text (tweets) for the task of **Named Entity Recognition**. Tweets are pre-processed and annotated as per the **6 NER tags** and a 7th  **Other** tag. 

### NER-Tags ###
- **B-Per** Indicates the Begening of a Person's name.
- **I-Per** Indicates the intermediate of a Person's name.
- **B-Org** Indicates the Begening of a Organizations's name.
- **I-Org** Indicates the intermediate of a Organizations's name.
- **B-Loc** Indicates the Begening of a Locations's name.
- **I-Loc** Indicates the intermediate of a Locations's name.
- **Other** Indicates all the word not falling in any of the above 6.

eg:

|    #Word    |    #Tag    |
|-------------|------------|
|    Bharat    |    B-Loc    |
|    ke    |    Other    |
|    2016    |    Other    |
|    ke    |    Other    |
|    Demonetization    |    Other    |
|    mein    |    Other    |
|    kitna    |    Other    |
|    kala    |    Other    |
|    dhan    |    Other    |
|    real    |    Other    |
|    mein    |    Other    |
|    aaya    |    Other    |
|    ???    |    Other    |
|    Accha    |    Other    |
|    hua    |    Other    |
|    ye    |    Other    |
|    prashna    |    Other    |
|    Miss    |    B-Per    |
|    Word    |    I-Per    |
|    Chillar    |    I-Per    |
|    ko    |    Other    |
|    nahi    |    Other    |
|    puccha    |    Other    |
|    gaya    |    Other    |
|    0    |    Other    |
|    #misschillar    |    B-Per    |
|    #missworld    |    Other    |
|    #Demonetisation    |    Other    |
|    #notebandi    |    Other    |
|    #modi    |    B-Per    |
|    #bjp    |    B-Org    |
|    #gujrat    |    B-Loc

---------------------------------------------------------------------------

### Contents ###
- `TwitterData` folder contains Id's of the scrapped tweets inside `Scrapped` folder, and processed and annotated data as named inside this.
- All the three Models.py are the files for the three ML classification models we used for our reserach paper.
- preprocessing and vector creation scripts are added with names indicating that.
- This dataset is in development and in future we will extend this to more number of tweets so as to make it a more reliable dataset for this taska and others.

----------------------------------------------------------------------------

### Outputs ###
- DecisionTree and CRF models have direct `score` calls that gives all the required stats.
- Keras does not provide the same for displaying score stats for LSTM model, so we build a coustom call of all the measure values and took average over all the iterations (here 5).
- All the models performed well on the given data.
- `Decision Tree` model with a **f1-score** of **0.94**.
- `Conditional Random Field (CRF)` model with a **f1-score** of **0.95**.
- `LSTM` model with a **f1-score** of **0.95**.

------------------------------------------------------------------------------
##### Authors #####
- Vinay Singh
- Deepanshu Vijay
- Syed A. Sarfaraz
- Manish Srivastava

LTRC
IIIT-Hyderabad

-------------------------------------------------------------------------------
##### Citation #####
**Named Entity Recognition for Hindi-English Code-Mixed Social Media Text**

2018, 27-35, Proceedings of the Seventh Named Entities Workshop [here](https://aclanthology.info/papers/W18-2405/w18-2405)
