'''
pathy = '/media/vol2/walid/work/github/semantic_searcher/'
conceptualizer = Conceptualizer()
conceptualizer.save(model=model, titles_path='/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/titles_redirects_ids.csv', outpath='/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/w2v-plain-anno-titles-4.4-10iter-dim500-wind9-cnt1-skipgram1.bin', vocab=None, concepts_only=False, log_every=1000000)
conceptualizer.load(model_path='/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/w2v-plain-anno-titles-4.4-10iter-dim500-wind9-cnt1-skipgram1.bin', log_every=1000000)
conceptualizer.load(model_path=pathy+'w2v-plain-anno-titles-4.4-10iter-dim500-wind9-cnt1-skipgram1.bin', log_every=1000000)
conceptualized = conceptualizer.conceptualize('Conversation is a form of interactive, spontaneous communication between two or more people. Typically, it occurs in spoken communication, as written exchanges are usually not referred to as conversations. The development of conversational skills and etiquette is an important part of socialization. The development of conversational skills in a new language is a frequent focus of language teaching and learning. Conversation analysis is a branch of sociology which studies the structure and organization of human interaction, with a more specific focus on conversational interaction.', ngram_range=(1,5),stop_words=list(sklearn.feature_extraction.stop_words.ENGLISH_STOP_WORDS))
all_titles_lower = get_all_titles(titles=titles, redirects=redirects, model=model)

out_vecs = conceptualize(conceptualizer,'universal congress of esperanto and garrick bar went to mixed incontinence new your times square ', ngram_range=(1,5))
for i in conceptualized:
    print(i[0]+'\t|\t'+i[1])
    

'''

'''
input = 'Conversation is a form of interactive, spontaneous communication between two or more people. Typically, it occurs in spoken communication, as written exchanges are usually not referred to as conversations. The development of conversational skills and etiquette is an important part of socialization. The development of conversational skills in a new language is a frequent focus of language teaching and learning. Conversation analysis is a branch of sociology which studies the structure and organization of human interaction, with a more specific focus on conversational interaction.'
vocabulary = ['new york', 'new', 'york', 'new york times', 'times', 'journal', 'of','in']
vectorizer = CountVectorizer(ngram_range=(1,3),vocabulary=vocabulary)
univectorizer = CountVectorizer(ngram_range=(1,1),vocabulary=vocabulary)
x = ['journal of hello new, york times']
x_transformed = vectorizer.fit_transform(x)
x_original = vectorizer.inverse_transform(x_transformed)
y

conceptualize test
all_titles = {'new york times':('123','New York Times'),'journal of':('432','Journal Of')}
titles = {'New York Times':'123','Journal Of':'432'}
model = {'id123di':'Vector New York Times','id432di':'Vector Journal Of'}
from nltk import ngrams
import re
token_pattern=r"(?u)\b\w\w+\b"
ngram_range = (1,5)
lowercase=True
input_text = 'journal of hello new, york times from here'
if lowercase==True:
    input_text = input_text.lower()

input_text = ' '.join(re.findall(token_pattern, input_text))
input_text_tokenized = input_text
all_vecs = []
for ngram_len in range(ngram_range[1],ngram_range[0]-1,-1):
    ngram_list = ngrams(input_text.split(), ngram_len)
    for ngram_token in ngram_list:
        ngram_token = ' '.join(ngram_token)
        title = all_titles.get(ngram_token) # check if a title, redirect
        if title is None: # not a title, redirect, may be a word
            title = ngram_token
            org_title = ''
        else:
            org_title = title[1]
            title = 'id'+titles[title[1]]+'di' # retrieve original title
        print(ngram_token,title)
        vec = model.get(title,'<UNK>')
        if vec!='<UNK>' or ngram_len==1: # append vector or unk if unigrams
            indx = '<TOK>' + str(len(all_vecs)) + '<TOK>'
            all_vecs.append((ngram_token,org_title,vec))
            input_text_tokenized = input_text_tokenized.replace(ngram_token,indx)

final_tokens = [int(i) for i in re.findall(r"<TOK>([0-9]+)<TOK>",input_text_tokenized)]
tokens_num = len(final_tokens)
out_vecs = [0]*tokens_num
for token_indx in range(tokens_num):
    out_vecs[token_indx] = all_vecs[final_tokens[token_indx]]

out_vecs
'''

'''
conceptualize fn
from collections import namedtuple
Concept_Ngram = namedtuple('Concept_Ngram', ['ngram','title','id','embeddings'])

def conceptualize(self, input_text='', ngram_range=(1,1),stop_words=set(nltk.corpus.stopwords.words('english')), token_pattern=r"(?u)\b\w\w+\b",lowercase=True):
    from nltk import ngrams
    import re
    import numpy as np
    if lowercase==True:
        input_text = input_text.lower()
    input_text = ' '.join(re.findall(token_pattern, input_text))
    all_vecs = []
    if stop_words is not None:
        stop_words_set = set(stop_words)
        # replace each stopword with space
        new_input_text = []
        for token in input_text.split(' '):
            if token in stop_words_set:
                new_input_text.append(' ')
            else:
                new_input_text.append(token)
        input_text = ' '.join(new_input_text)
    input_text = ' ' + input_text + ' '    # replace('or','<tok>1<tok>') would result in more --> m<tok>1<tok>
    input_text_tokenized = input_text
    for ngram_len in range(ngram_range[1],ngram_range[0]-1,-1):
        ngram_list = ngrams(input_text.split(' '), ngram_len)
        for ngram_token in ngram_list:
            if '' in ngram_token: # this ngram is not correct, original string had a stopword
                continue
            ngram_token = ' '.join(ngram_token)
            title = self.all_titles.get(ngram_token) # check if a title, redirect
            if title is None: # not a title, redirect, may be a word
                title = ngram_token
                org_title = ''
                id = ''
            else:
                org_title = title[1]
                title = title[0] # retrieve original title
                id = title.replace('id','').replace('di','')
            vec = self.model.get(title,'<UNK>')
            if isinstance(vec,np.ndarray) or (vec=='<UNK>' and ngram_len==1): # append vector or <unk> if unigrams
                indx = ' '+'<TOK>' + str(len(all_vecs)) + '<TOK>'+' '
                all_vecs.append(Concept_Ngram(ngram_token,org_title,id,vec))
                input_text_tokenized = input_text_tokenized.replace(' '+ngram_token+' ',indx)
    final_tokens = [int(i) for i in re.findall(r"<TOK>([0-9]+)<TOK>",input_text_tokenized)]
    tokens_num = len(final_tokens)
    out_vecs = [0]*tokens_num
    for token_indx in range(tokens_num):
        out_vecs[token_indx] = all_vecs[final_tokens[token_indx]]
    return out_vecs





conceptualized = conceptualize(conceptualizer,'Conversation is a sssosss form of interactive, spontaneous communication between two or more people.', ngram_range=(1,5))
for i in conceptualized:
    print(i[0]+'\t|\t'+i[1])

conceptualized = conceptualize(conceptualizer,'Conversation is a form of interactive, spontaneous communication between two or more people. Typically, it occurs in spoken communication, as written exchanges are usually not referred to as conversations. The development of conversational skills and etiquette is an important part of socialization. The development of conversational skills in a new language is a frequent focus of language teaching and learning. Conversation analysis is a branch of sociology which studies the structure and organization of human interaction, with a more specific focus on conversational interaction.', ngram_range=(1,5))
'''