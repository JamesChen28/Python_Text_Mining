
文章(Document)：一段文字  
語料庫(Corpus)：Document物件的集合，作為Model的輸入  
向量(Vector)：文字特徵構成的列表  
稀疏向量(Sparse Vector)：去除向量多餘的0  
詞袋向量(Bag-of-words Vector)：BOW，將文字轉成向量  

模型(Model)：兩個向量空間的變換
詞袋模型(Bag-of-words Model)：不考慮文法以及詞的順序的模型
tf-idf Model：BOW轉換成Vector Space，根據稀有度加權  


預處理
1. 文章轉語料庫  

i. 分詞 nltk.tokenize.RegexpTokenizer / nltk.word_tokenize  
ii. 移除停止詞 Stop word nltk.corpus.stopwords  
iii. 標註詞性 nltk.pos_tag  
iv. 字幹搜尋 Stemming nltk.stem.PorterStemmer  
v. 詞性還原 Lemmatization nltk.stem.WordNetLemmatizer  

texts = []  
for i in range(len(Document)):  
    tokenizer = RegexpTokenizer(r'\w+')  
    tokens = tokenizer.tokenize(Document[i])  
    texts.append(tokens)  


2. 建立字典，轉成向量  

import gensim.corpora  
dictionary = corpora.Dictionary(texts)  
corpus = [dictionary.doc2bow(text) for text in texts]  


3. 主題向量轉換(Transformation)  

import gensim.models  
向量變換對應著主題模型的轉換  

i. TF-IDF(Term Frequency Inverse Document Frequency)  
tfidf = models.TfidfModel(corpus)  

corpus_tfidf = tfidf[corpus]  
for doc in corpus_tfidf:  
    print(doc)  

doc_bow = [(0, 1), (1, 1)]  
tfidf[doc_bow]  


ii. LSI/LSA(Latent Semantic Indexing)  
lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)  
corpus_lsi = lsi_model[corpus_tfidf]  

model.add_documents(another_corpus_tfidf)  
lsi_vec = model[tfidf_vec]  


iii. RP(Random Projections)  
rp_model = models.RpModel(corpus_tfidf, num_topics=500)


iv. LDA(Latent Dirichlet Allocation)  
lda_model = models.LdaModel(corpus, id2word=dictionary, num_topics=100)


v. HDP(Hierarchical Dirichlet Process)  
hdp_model = models.HdpModel(corpus, id2word=dictionary)


vi. pLSA(Probabilistic Latent Semantic Analysis)  


vii. lda2vec   


4. 相似度查詢  

import gensim.similarities  
index = similarities.MatrixSimilarity(lsi_model[corpus])  
index.save('tmp/deerwester.index')  

doc = 'Human computer interaction'  
vec_bow = dictionary.doc2bow(doc.lower().split())  
vec_lsi = lsi_model[vec_bow]   
index = similarities.MatrixSimilarity.load('tmp/deerwester.index')  

sims = index[vec_lsi]  
print(list(enumerate(sims)))


5. Word2Vec  


