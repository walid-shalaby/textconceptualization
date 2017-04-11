"""
tester for Conceptualizer class
"""


import argparse
from conceptualizer import Conceptualizer

#sample input text: Concept learning, also known as category learning, concept attainment, and concept formation, is largely based on the works of the cognitive psychologist Jerome Bruner. Bruner, Goodnow, & Austin (1967) defined concept attainment (or concept learning) as "the search for and listing of attributes that can be used to distinguish exemplars from non exemplars of various categories." More simply put, concepts are the mental categories that help us classify objects, events, or ideas, building on the understanding that each object, event, or idea has a set of common relevant features. Thus, concept learning is a strategy which requires a learner to compare and contrast groups or categories that contain concept-relevant features with groups or categories that do not contain concept-relevant features.
#python3 conceptualizer_test.py --model_path /media/vol2/walid/work/github/semantic_searcher/w2v-plain-anno-titles-4.4-10iter-dim500-wind9-cnt1-skipgram1.bin
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', dest='model_path', required=True,nargs='?', type=str, default='')
FLAGS = parser.parse_args()
print('model_path=',FLAGS.model_path)
conceptualizer = Conceptualizer()
conceptualizer.load(model_path=FLAGS.model_path, log_every=1000000)
input_text = input('Enter text (enter to skip): ')
while len(input_text)>0:
    min_ngrams = int(input('Enter min ngrams: '))
    max_ngrams = int(input('Enter max ngrams: '))
    conceptualized = conceptualizer.conceptualize(input_text=input_text, ngram_range=(min_ngrams,max_ngrams))
    if input('print tokens? (y/n) ')=='y':
        print('tokens',[concept.ngram for concept in conceptualized])
    if input('print article titles? (y/n) ')=='y':
        print('titles',[concept.title for concept in conceptualized])
    if input('print article ids? (y/n) ')=='y':
        print('id',[concept.id for concept in conceptualized])
    if input('print embeddings? (y/n) ')=='y':
        print('embeddings',[concept.embeddings for concept in conceptualized])
    input_text = input('Enter text (enter to skip): ')

print('done!')