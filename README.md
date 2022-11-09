## Outliers in Text Corpora
- usually text corpora contain a mix of exhaustive as well as non-exhaustive documents
- such a mix of wide genres and themes of documents extensively increase the occurances of anomalies, which need to be flagged for deployed ML systems
- this notebook studies and implements the same to analyze anomalous texts and emerging themes in large NLP corpora 
- the workflow is based on: Transformers, cleablab, UMAP, and c-TF-IDF from BERTopic
    1. `AutoTokenizer` and `AutoModel` submodules from `transformers` are used to obtain accurate tokenized representations from the raw text belonging to different genres and themes and retrieve the relevant model architecture, using a pretrained tokenizer+model from HuggingFaceHub
    2. PyTorch datasets are obtained from preprocessed text datasets from HuggingFaceHub, and operations are performed over the dataset using `torch`
    3. `OutOfDistribution` submodule of `cleanlab` is used to determine the outliers based on outlier scores using a nearest neighbour estimator
    4. the results are further processed and visualized to conclude the anomaly trends using `UMAP` and class based `TF-IDF` to further filter out the outliers using the weights of words/phrased in the corpus
    5. once the anomalous genres are clustered, topics within them are identified depicting out-of-distribution topics
