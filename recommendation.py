from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re

# 데이터 텍스트 전처리
def preprocess_text(text):
    # 불필요한 문자 및 공백 제거
    text = re.sub(r'[^A-Za-z0-9가-힣\s]', '', text)
    # 소문자 변환
    text = text.lower()
    # 불용어 제거
    stop_words = ['the', 'and', 'in', 'of', 'a', 'an', 'to']  # 불용어 목록을 필요에 따라 수정
    text = ' '.join(word for word in text.split() if word not in stop_words)

    return text

def preprocess_input(data, name, comp):
    name = preprocess_text(name)
    # 'data' 데이터프레임에 제품명이 이미 있는지 확인
    if name and name in data['제품명'].values:
        selected_additive = data[data['제품명'].str.contains(name, case=False)]['첨가제'].values[0]
    else:
        # 제품명이 'data' 데이터프레임에 없는 경우, 새로운 행 추가
        tmp = pd.DataFrame({'제품명': name, '첨가제': comp}, index=[0])
        data = pd.concat([data, tmp], ignore_index=True)
        #data = data.append({'제품명': name, '첨가제': comp}, ignore_index=True)
        #selected_additive = data[data['제품명'].str.contains(user_input, case=False)]['첨가제'].values[0]
    return name, data

def recommend(name, comp):
    print(comp)
    df = pd.read_csv('data/product_data.csv')       
    data = df[['제품명', '업체명', '첨가제', '전문의약품', '제조/수입']]

    name, data = preprocess_input(data, name, comp)
    #print(name)
    # 데이터프레임 'data'에 '첨가제' 열을 전처리
    data['첨가제'] = data['첨가제'].apply(preprocess_text)

    # TF-IDF 벡터화
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    additives_tfidf = tfidf_vectorizer.fit_transform(data['첨가제'])

    # 이미지에서 추출한 첨가제와 'data' 데이터프레임의 '첨가제' 정보 간의 유사도 계산 (cos 유사도)
    similarities = cosine_similarity(additives_tfidf)

    # 유사도를 데이터프레임으로 변환
    similarity_df = pd.DataFrame(similarities, columns=data['제품명'], index=data['제품명'])

    # 입력된 첨가제와 유사도가 높은 순으로 추천 제품을 선택
    similar_products = similarity_df[name].sort_values(ascending=False)[1:6]

    # 추천 결과
    return similar_products