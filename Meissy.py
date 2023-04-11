import konlpy
from konlpy.tag import Okt
import re
from pykospacing import Spacing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import seaborn as sns
from keras.utils import to_categorical

#특수문자 처리
def clean_str(text):
    pattern = '([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)' # E-mail제거
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = '(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+' # URL제거
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = '([ㄱ-ㅎㅏ-ㅣ]+)'  # 한글 자음, 모음 제거
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = '<[^>]*>'         # HTML 태그 제거
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = '[^\w\s\n]'         # 특수기호제거
    text = re.sub(pattern=pattern, repl='', string=text)
    text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]','', string=text)
    text = re.sub('\n', '.', string=text)
    # text = text.replace(' ','')
    return text 

#int, str, 혼합 구분
def decide_kind(text) :
  flag = True
  str_flag = False
  int_flag = False

  for var in text:
    try:
      int(var)
      int_flag = True
    except ValueError:
      str_flag = True

  if int_flag and str_flag :
    return 2
  elif int_flag :
    return 1
  else :
    return 0
  
#임시 데이터 generation 더 많은 데이터를 생성할 수 있는 방법을 생각해야 할듯
# Program to generate a random number between 0 and 9
# importing the random module
import random

#날짜 값
for i in range(1,13) : 
  date_words.append(str(i))
  date_words.append(str(i) + "월")
  for j in range(1,32) :
    date_words.append(str(j))
    date_words.append(str(i) + "월" + str(j) + "일")

for i in range(1, 100) :
  phone_number.append(['','010' + str(random.randint(9999999,100000000))])

for i in range(1, 100) :
  phone_number.append(['',str(random.randint(9999999,100000000))])
  
 #테스트 데이터 생성
data = []

for op in operators :
  np_op = np.array(op)
  section_name = clean_str(str(np_op[0]))
  data.append([section_name, len(section_name), decide_kind(section_name), 'section'])
  for op_name in np_op[1:] :
    operator_name = clean_str(op_name)
    last_two_name = clean_str(op_name[1:])
    data.append([clean_str(op_name),len(operator_name), decide_kind(operator_name), 'name'])
    data.append([last_two_name,len(last_two_name), decide_kind(last_two_name), 'name'])


for number in phone_number :
  number_name = number[1]
  last_four_number = number[1][-4:]
  data.append([number_name, len(number_name),decide_kind(number_name), 'number'])
  data.append([last_four_number, 4, decide_kind(last_four_number), 'number'])

for date_word in date_words :
  date_name = clean_str(date_word)
  data.append([date_name, len(date_name),decide_kind(date_name), 'date'])


df = pd.DataFrame(data,columns = ['input','inputLength','type','class'])

df.isnull().values.any() #결측값체크
df['input'].replace('',np.nan,inplace=True)
df['class'].replace('',np.nan,inplace=True)
df.drop_duplicates(subset=['input'], inplace=True) #중복제거
df.dropna(subset=['input'], inplace=True) #중복제거


def getClassEncoding(string) : 
  if string == "name" : 
    return 0
  elif string == "section" :
    return 1
  elif string == "number" :
    return 2
  else :
    return 3

def getInputEncoding(string) :
  test = 0
  for name in string :
    test += ord(name)
  return test

df['raw'] = df.apply(lambda x : x['input'], axis=1 )
df['class'] = df.apply(lambda x : getClassEncoding(x['class']), axis=1)
df['input'] = df.apply(lambda x : getInputEncoding(x['input']), axis=1 )

#데이터 전처리
def get_morphs(input_text) :
  #띄어쓰기
  spacing = Spacing()
  inputText = spacing(input_text)

  print(inputText)

  #형태소 끊기
  okt = Okt()
  nouns = okt.morphs(inputText)

  print("nons : ",nouns)
  #stop_words처리
  stop_words = ["검색", "찾아", "찾아줘"]

  #한자리 이하 제거
  preprocessed_words = []
  for noun in nouns :
    if (noun in stop_words) :
      return
    if len(noun) > 1 :
      preprocessed_words.append(noun)

  return preprocessed_words

#모델 구현
count_ratio = df.groupby('class').size().reset_index(name='count')
is_name_ratio = round(df["class"].value_counts()[0] / len(data) * 100, 3)
is_section_ratio = round(df["class"].value_counts()[1] / len(data) * 100, 3)
is_number_ratio = round(df["class"].value_counts()[2] / len(data) * 100, 3)
is_date_ratio = round(df["class"].value_counts()[3] / len(data) * 100, 3)

print(count_ratio)
print("데이터 중 이름이 차지하는 비율 : ",is_name_ratio)
print("데이터 중 섹션이 차지하는 비율 : ",is_section_ratio)
print("데이터 중 숫자 차지하는 비율 : ",is_number_ratio)
print("데이터 중 날짜가 차지하는 비율 : ",is_date_ratio)

data_X = df[['input','inputLength','type']].values
data_y = df['class'].values

print(data_X[:5])
print(data_y[:5])

(X_train, X_test, y_train, y_test) = train_test_split(data_X, data_y, train_size=0.7, random_state=1)

# 원-핫 인코딩
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train[:5])
print(y_test[:5])

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(4, input_dim=3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=200, batch_size=1, validation_data=(X_test, y_test))

#손실율 확인
epochs = range(1, len(history.history['accuracy']) + 1)
plt.plot(epochs, history.history['loss'])
plt.plot(epochs, history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

#활용
preprocessed_words = get_morphs(clean_str("01027403096 추새벽 전화번호 맞아?"))
for word in preprocessed_words :
  input_name = clean_str(word);
  input_type = decide_kind(input_name)
  input_length = len(input_name)
  text_input = getInputEncoding(input_name)

  matrix = [text_input, input_length, input_type]
  result = model.predict([matrix])
  array = np.array(result[0])

  print(input_name)
  print(matrix)
  print(result)
