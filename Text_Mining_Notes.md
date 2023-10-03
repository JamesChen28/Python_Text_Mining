
ref  
https://hackmd.io/@gensimDART/SydKygQa_
https://zhuanlan.zhihu.com/p/37175253  
文章(Document)：一段文字  
語料庫(Corpus)：Document物件的集合  
向量(Vector)：文字特徵構成的列表  
稀疏向量(Sparse Vector)：去除向量多餘的0  
模型(Model)：兩個向量空間的變換

預處理
1. 文章轉語料庫  
i. 分詞 nltk.tokenize.RegexpTokenizer / nltk.word_tokenize  
ii. 移除停止詞 nltk.corpus.stopwords  

texts = []  
for i in range(len(Document)):  
    tokenizer = RegexpTokenizer(r'\w+')  
    tokens = tokenizer.tokenize(Document[i])  
    texts.append(tokens)  

2. 建立字典，轉成向量 gensim.corpora  
dictionary = corpora.Dictionary(texts)  
corpus = [dictionary.doc2bow(text) for text in texts]

3. 建立模型  
i. TF-IDF gensim.models  
tfidf = models.TfidfModel(corpus)  
corpus_tfidf = tfidf[corpus]  
for doc in corpus_tfidf:  
    print(doc)  

ii. LSI/LSA  
lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)  
corpus_lsi = lsi_model[corpus_tfidf]  

model.add_documents(another_tfidf_corpus)  
lsi_vec = model[tfidf_vec]  

iii. RP  


iv. LDA  


v. HDP  

4. 相似度查詢 gensim.similarities  
doc = "Human computer interaction"  
vec_bow = dictionary.doc2bow(doc.lower().split())  
vec_lsi = lsi_model[vec_bow]   
index = similarities.MatrixSimilarity(lsi_model[corpus])
index.save('tmp/deerwester.index')
index = similarities.MatrixSimilarity.load('tmp/deerwester.index')

sims = index[vec_lsi]  # perform a similarity query against the corpus
print(list(enumerate(sims)))

5. Word2Vec




https://dysonma.github.io/2020/12/12/NLP%E6%96%B7%E8%A9%9E%E7%B5%B1%E8%A8%88%E5%88%86%E6%9E%90-II-NLTK%E3%80%81wordnet/  
nltk statistics

https://www.datacamp.com/tutorial/discovering-hidden-topics-python  
Comparison Between Text Classification and Topic Modeling

https://medium.com/@fredericklee_73485/%E4%BD%BF%E7%94%A8lsa-plsa-lda%E5%92%8Clda2vec%E9%80%B2%E8%A1%8C%E5%BB%BA%E6%A8%A1-7ab56c18164a  
LSA PLSA LDA lda2Vec

