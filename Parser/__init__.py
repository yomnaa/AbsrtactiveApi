from bs4 import BeautifulSoup
try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen
import tldextract


def parse(url):
    extracted = tldextract.extract(url)
    netloc = "{}.{}".format(extracted.domain, extracted.suffix)
    if netloc == "cnn.com":
        return textCnn(url)


def textCnn(url):
    final_text = ""
    try:
        article = urlopen(url)
        soup = BeautifulSoup(article, 'html.parser')
        name_box = soup.findAll(attrs={'class': 'zn-body__paragraph'})
        for child in name_box:
            final_text += " " + child.text
        return final_text
    except:
        return ""
