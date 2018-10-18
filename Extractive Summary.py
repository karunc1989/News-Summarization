# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 09:33:43 2018

@author: I354298
"""
import nltk
import pickle
nltk.download('punkt')
nltk.download('stopwords')
import glob
import os
import spacy
import regex
import re
import pprint
import string
import heapq

    
Contractions_Dict = {
  "ain't": "am not",
  "aren't": "are not",
  "can't": "cannot",
  "can't've": "cannot have",
  "'cause": "because",
  "could've": "could have",
  "couldn't": "could not",
  "couldn't've": "could not have",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "hadn't": "had not",
  "hadn't've": "had not have",
  "hasn't": "has not",
  "haven't": "have not",
  "he'd": "he would",
  "he'd've": "he would have",
  "he'll": "he will",
  "he'll've": "he will have",
  "he's": "he is",
  "how'd": "how did",
  "how'd'y": "how do you",
  "how'll": "how will",
  "how's": "how is",
  "I'd": "I would",
  "I'd've": "I would have",
  "I'll": "I will",
  "I'll've": "I will have",
  "I'm": "I am",
  "I've": "I have",
  "isn't": "is not",
  "it'd": "it had",
  "it'd've": "it would have",
  "it'll": "it will",
  "it'll've": "it will have",
  "it's": "it is",
  "let's": "let us",
  "ma'am": "madam",
  "mayn't": "may not",
  "might've": "might have",
  "mightn't": "might not",
  "mightn't've": "might not have",
  "must've": "must have",
  "mustn't": "must not",
  "mustn't've": "must not have",
  "needn't": "need not",
  "needn't've": "need not have",
  "o'clock": "of the clock",
  "oughtn't": "ought not",
  "oughtn't've": "ought not have",
  "shan't": "shall not",
  "sha'n't": "shall not",
  "shan't've": "shall not have",
  "she'd": "she would",
  "she'd've": "she would have",
  "she'll": "she will",
  "she'll've": "she will have",
  "she's": "she is",
  "should've": "should have",
  "shouldn't": "should not",
  "shouldn't've": "should not have",
  "so've": "so have",
  "so's": "so is",
  "that'd": "that would",
  "that'd've": "that would have",
  "that's": "that is",
  "there'd": "there had",
  "there'd've": "there would have",
  "there's": "there is",
  "they'd": "they would",
  "they'd've": "they would have",
  "they'll": "they will",
  "they'll've": "they will have",
  "they're": "they are",
  "they've": "they have",
  "to've": "to have",
  "wasn't": "was not",
  "we'd": "we had",
  "we'd've": "we would have",
  "we'll": "we will",
  "we'll've": "we will have",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "what'll": "what will",
  "what'll've": "what will have",
  "what're": "what are",
  "what's": "what is",
  "what've": "what have",
  "when's": "when is",
  "when've": "when have",
  "where'd": "where did",
  "where's": "where is",
  "where've": "where have",
  "who'll": "who will",
  "who'll've": "who will have",
  "who's": "who is",
  "who've": "who have",
  "why's": "why is",
  "why've": "why have",
  "will've": "will have",
  "won't": "will not",
  "won't've": "will not have",
  "would've": "would have",
  "wouldn't": "would not",
  "wouldn't've": "would not have",
  "y'all": "you all",
  "y'alls": "you alls",
  "y'all'd": "you all would",
  "y'all'd've": "you all would have",
  "y'all're": "you all are",
  "y'all've": "you all have",
  "you'd": "you had",
  "you'd've": "you would have",
  "you'll": "you you will",
  "you'll've": "you you will have",
  "you're": "you are",
  "you've": "you have"
}




# Importing the Corpus in Python
#corpusdir = r"C:\Users\I354298\Desktop\AIML\cnn_stories\cnn\stories\*.txt"              #Corpus data directory
#file_list = glob.glob(corpusdir)
#News_Corpus =[]
#for path in file_list:
#    with open(path, encoding = "utf-8") as fpath:
#        News_Corpus.append(fpath.read())
        
        

       

#Spilt the highlights from the news story
def News_splitting(article):
    splitat = article.find('@highlight')
    Story = article[:splitat]
    Highlight = article[splitat:].split('@highlight')
    return Story,Highlight

###  Text Normalization    

# Remove Special Characters
def Removing_Special_Character(sentence):
    var1 = r'[?|$|*|(|)~|@|%|&|-|>>|]'
    var2 = r'[\n|-]'
 #   var3 = r"\W"
    Filtered_sentence1 = re.sub(var1,r' ',sentence)
    Filtered_sentence2 = re.sub(var2,r'',Filtered_sentence1)
  #  Filtered_sentence3 = re.sub(var3,r" ",sentence)
    return Filtered_sentence2


#Removing CNN and story location

def Remove_CNN(article):
    index = article.find('CNN')
    if index >-1:
        story = article[index+len('CNN'):]
    else:
        story = article 
    return story

#Replace Contractions
def expand_contractions(sentence, contraction_mapping):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
        if contraction_mapping.get(match)\
        else contraction_mapping.get(match.lower())
        if expanded_contraction != None:
            expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
    expanded_sentence = contractions_pattern.sub(expand_match, sentence)
    return expanded_sentence


# Text Tokenization

def sent_token(data):
    sentences = nltk.sent_tokenize(data)
    return sentences

def Extraction_prep(sent_token,article_clean):
    list_stopwords = ['i',"'s", 'me',',', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    word2count = {}
    for word in nltk.word_tokenize(article_clean.lower()):     
        if word not in list_stopwords:                  
            if word not in word2count.keys():
                word2count[word]=1
            else:
                word2count[word]+=1               
    sent2score = {}
    for sentence in sent_token:
        for word in nltk.word_tokenize(sentence.lower()):
            if word in word2count.keys():
                if len(sentence.split(' '))<60:
                    if sentence not in sent2score.keys():
                        sent2score[sentence]=word2count[word]
                    else:
                        sent2score[sentence]+=word2count[word]
    return word2count, sent2score

def Cleaning_article(article):
    article_clean,article_highlight = News_splitting(article) 
    article_clean = Removing_Special_Character(article_clean)
    article_clean = Remove_CNN(article_clean)
    article_clean = expand_contractions(article_clean,Contractions_Dict)
    sent_tokens   = sent_token(article_clean)
    return article_clean,sent_tokens

def Extraction(article):   
    article_clean,sent_token = Cleaning_article(article)
    word_score , sent_score = Extraction_prep(sent_token,article_clean)
    best_sentences = heapq.nlargest(5,sent_score,key=sent_score.get)
    for sentences in best_sentences:
        print(sentences,'\n')















#Importing the article
path = r"C:\Users\I354298\Desktop\AIML\cnn_stories\cnn\stories\56893eed4689fb0f4be85d8b4961f2c022dafc25.story" 
with open(path, encoding = "utf-8") as fpath:
    article = fpath.read() 
Extraction(article)
