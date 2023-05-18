import re
import konlpy
from konlpy.tag import Okt
import numpy as np
import matplotlib.pyplot as plt
import urllib.request


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from soyspacing.countbase import CountSpace


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional, TimeDistributed
from tensorflow.keras.optimizers import Adam


#dictionary 만들기
def make_predefined_dictionaries() :
  stopwords = open("/home/ubuntu/python_test/library/stopwords.txt","r")
  #stopwords = open("/content/saebyeok/stopwords.txt","r")
  stopwords = stopwords.readlines()


  dic_stopword = [stopword.replace('\n','') for stopword in stopwords if stopword.startswith('#') == False and len(stopword.strip()) > 0]


  names = open("/home/ubuntu/python_test/library/name.train.txt","r")
  #names = open("/content/saebyeok/name.train.txt","r")
  names = names.readlines()


  dic_names = [name.replace('\n','').split()[0] for name in names if name.startswith('#') == False and len(name.strip()) > 0]


  chains = open("/home/ubuntu/python_test/library/chain.train.txt","r")
  #chains = open("/content/saebyeok/chain.train.txt","r")
  chains = chains.readlines()


  dic_chains = [chain.replace('\n','').split()[0] for chain in chains if chain.startswith('#') == False and len(chain.strip()) > 0]


  dic_phone_numbers = open("/home/ubuntu/python_test/library/phone.train.txt","r")
  #dic_phone_numbers = open("/content/saebyeok/phone.train.txt","r")
  dic_phone_numbers = dic_phone_numbers.readlines()


  dic_phone_numbers = [phone_number.replace('\n','').split()[0] for phone_number in dic_phone_numbers if phone_number.startswith('#') == False and len(phone_number.strip()) > 0]


  return [dic_stopword, dic_names, dic_chains, dic_phone_numbers]


#전처리
def spacing(text) :
  spacing_model = CountSpace()
  spacing_model.load_model("/home/ubuntu/python_test/model/spacing",json_format=False)
  corrected_sentence, tag = spacing_model.correct(text)


  return corrected_sentence
  
def clean(text) :
  pattern = '([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)'
  text = re.sub(pattern=pattern, repl='', string=text)
  pattern= '(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
  text = re.sub(pattern=pattern, repl='', string=text)
  pattern = '([ㄱ-ㅎㅏ-ㅣ]+)'
  text = re.sub(pattern=pattern, repl='', string=text)
  pattern = '<[^>]*>'
  text = re.sub(pattern=pattern, repl='', string=text)
  pattern = '[^\w\s\n]'
  text = re.sub(pattern=pattern, repl='', string=text)


  return text


def get_morphs(text) :
  dic_stopword, dic_names, dic_chains, dic_phone_numbers = make_predefined_dictionaries()


  text = clean(text)
  text = spacing(text)


  okt = Okt()
  nouns = okt.morphs(text)


  #걸러내기 : 한글자 이하 / 욕설, 성적인 언급 등 제외
  preprocessed_words=[]
  for noun in nouns :
      if len(noun) > 1 and noun not in dic_stopword :
          preprocessed_words.append(noun)


  return preprocessed_words
  
def rule_based(nouns) :
  dic_stopword, dic_names, dic_chains, dic_phone_numbers = make_predefined_dictionaries()


  rule_result = []
  remove_target = []


  #주어진 명사가 두글자에서 세글자사이라면 이름을 체크
  for noun in nouns :
    if len(noun) <= 3 :
      r = re.compile(".*{0}.*".format(noun))
      matched_name = list(filter(r.match,dic_names))
      if len(matched_name) > 0 :
        rule_result.append({'entity' : 'B_PS', 'value' : matched_name[0]})
        remove_target.append(noun)


  for target in remove_target :
    nouns.remove(target)




  #체인으로 체크
  remove_target = []


  for noun in nouns : 
    r = re.compile(".*{0}.*".format(noun.upper()))
    matched_chain = list(filter(r.match,dic_chains))
    if len(matched_chain) > 0 :
      rule_result.append({'entity' : 'B_OG', 'value' : matched_chain[0]})
      remove_target.append(noun)
  
  for target in remove_target :
    nouns.remove(target)




  #전화번호와 년도가 잘 되지 않는 군
  #숫자로만 이루어져 있으면 전화번호로 분류
  remove_target = []


  for noun in nouns : 
    if noun.isnumeric():
      rule_result.append({'entity' : 'B_PN', 'value' : noun})
      remove_target.append(noun)
  for target in remove_target :
    nouns.remove(target)




  #월, 일, 년이 들어가 있으면 날짜로 분
  remove_target = []


  for noun in nouns : 
    if re.match('.*[월|일|년].*', noun):
      rule_result.append({'entity' : 'B_DT', 'value' : noun})
      remove_target.append(noun)


  for target in remove_target :
    nouns.remove(target)


  return rule_result


#모델 학습
def train() :
  tagged_sentences =[]
  sentence = []


  sentences = open("/home/ubuntu/python_test/library/ner_dataset.txt","r")
  #sentences = open("/content/saebyeok/ner_dataset.txt","r")
  sentences = sentences.readlines()


  for s in sentences :
    if s.startswith(';') or s.startswith('$') or s.startswith('#') or len(s.strip())==0 :
      if len(sentence) > 0 :
        tagged_sentences.append(sentence)
        sentence = []
      continue


    s.replace('\n','')
    s = s.split('\t')
    if len(s) != 4 :
      continue


    sentence.append([s[1].lower(),s[3].replace('\n','')])


  sentences, ner_tags = [],[]


  for tagged_sentence in tagged_sentences :
      sen, tag_info = zip(*tagged_sentence)
      sentences.append(list(sen))
      ner_tags.append(list(tag_info))




  vocab_size = len(sentences)
  src_tokenizer = Tokenizer(num_words = vocab_size, oov_token='OOV')
  src_tokenizer.fit_on_texts(sentences)


  tar_tokenizer = Tokenizer()
  tar_tokenizer.fit_on_texts(ner_tags)


  tag_size = len(tar_tokenizer.word_index) + 1


  X_train = src_tokenizer.texts_to_sequences(sentences)
  y_train = tar_tokenizer.texts_to_sequences(ner_tags)


  index_to_word = src_tokenizer.index_word
  index_to_ner = tar_tokenizer.index_word


  decoded = []
  for index in X_train[0] :
    decoded.append(index_to_word[index])


  max_len=70
  X_train = pad_sequences(X_train, padding='post', maxlen=max_len)
  y_train = pad_sequences(y_train, padding='post', maxlen=max_len)


  X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=.2, random_state=777)


  y_train = to_categorical(y_train, num_classes = tag_size)
  y_test = to_categorical(y_test, num_classes = tag_size)


  embedding_dim = 128
  hidden_units = 128


  model = Sequential()
  model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len, mask_zero=True))
  model.add(Bidirectional(LSTM(hidden_units, return_sequences=True)))
  model.add(TimeDistributed(Dense(tag_size, activation='softmax')))


  model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
  model.fit(X_train,y_train,batch_size=128,epochs=8,validation_data=(X_test,y_test))


  model.save('/home/ubuntu/python_test/meissy0.02')
  #model.save('/content/saebyeok/meissy0.02')


def predict(text) :
  from tensorflow import keras


  index_to_ner = {1: 'o', 2: 'i', 3: 'b_dt', 4: 'b_og', 5: 'b_ps', 6: 'b_lc', 7: 'b_ti'}


  rule_based_result = []
  predict_result = []


  #rule based
  nouns = get_morphs(text)
  rule_based_result = rule_based(nouns)


  if len(nouns) == 0 :
    return rule_based_result


  tagged_sentences =[]
  sentence = []


  sentences = open("/home/ubuntu/python_test/library/ner_dataset.txt","r")
  #sentences = open("/content/saebyeok/ner_dataset.txt","r")
  sentences = sentences.readlines()


  for s in sentences :
    if s.startswith(';') or s.startswith('$') or s.startswith('#') or len(s.strip())==0 :
      if len(sentence) > 0 :
        tagged_sentences.append(sentence)
        sentence = []
      continue


    s.replace('\n','')
    s = s.split('\t')
    if len(s) != 4 :
      continue


    sentence.append([s[1].lower(),s[3].replace('\n','')])


  sentences = []


  for tagged_sentence in tagged_sentences :
      sen, tag_info = zip(*tagged_sentence)
      sentences.append(list(sen))


  vocab_size = len(sentences)
  src_tokenizer = Tokenizer(num_words = vocab_size, oov_token='OOV')
  src_tokenizer.fit_on_texts(sentences)




  predict_target = src_tokenizer.texts_to_sequences([nouns])
  predict_target = pad_sequences( predict_target, padding='post', maxlen=70)


  loded_model = keras.models.load_model('/home/ubuntu/python_test/meissy0.02')
  y_predicted = loded_model.predict(np.array([predict_target[0]]))
  y_predicted = np.argmax(y_predicted,axis=-1)
  
  index = 0
  for word, pred in zip( predict_target[0], y_predicted[0]) :
    if word != 0 :
      predict_result.append({"entity" : index_to_ner[pred].upper(), "value" : nouns[index]})
    index = index + 1
  
  return predict_result + rule_based_result


