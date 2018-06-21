from nltk.tokenize.stanford import StanfordTokenizer
import Preprocessing
import Summarizer
def getModelApi():
    preprocessor = Preprocessing.Preprocessor()
    summarizer = Summarizer.Summarizer('./vocab', './')
    def preprocessorApi(article):
        tokenized=preprocessor.tokenize(article)
        tokenized=(' '.join(tokenized))
        return preprocessor.adjust_article(tokenized.split('*N*')).encode('utf-8')

    def modelApi(preprocessed_articles):
        return summarizer.summarize(preprocessed_articles)

    return preprocessorApi,modelApi


