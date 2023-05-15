#!/usr/bin/env python
# coding: utf-8

# In[340]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='whitegrid', font_scale=1.5)
sns.set_palette('Set2', n_colors=10)
plt.rc('font', family='AppleGothic')
plt.rc('axes', unicode_minus=False)

import streamlit as st
from datetime import date


# In[341]:


df1 = pd.read_csv('final2.csv',encoding = 'cp949')
qwe = pd.read_excel('폐업_count.xlsx',sheet_name = None)
locals().update(qwe)


# In[342]:


전체폐업 = 전체폐업.set_index('year')
구별폐업 = 구별폐업.set_index('year')


# In[343]:


df1['인허가일자'] = pd.to_datetime(df1['인허가일자'], format= '%Y-%m-%d')
df1['폐업일자'] = pd.to_datetime(df1['폐업일자'], format= '%Y-%m-%d')


# In[344]:


st.set_page_config(page_title = '폐업 Dashboard',
                  page_icon = '🍚', layout = 'wide')
st.title('Date App Dashboard')
if st.button('새로고침'):
    st.experimental_rerun()


# In[345]:


my_df1 = df1.copy()
st.sidebar.title("조건 필터")
st.sidebar.header("날짜 조건")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("시작일시", date(2018, 1, 1),
                                       min_value=date(2018,1,1),
                                       max_value=date(2021,12,30))
with col2:
    end_date = st.date_input("종료일시", date(2021, 12, 31),
                                     min_value=date(2019,1,2),
                                     max_value=date(2021,12,31))
my_df1 = my_df1[my_df1['인허가일자'].dt.date.between(start_date, end_date)]

option01 = st.sidebar.checkbox('영업중제외', value = False)
if option01:
    my_df1 = my_df1[my_df1['영업상태코드']!= 1]
    
st.sidebar.header('✈️지역분류')
option02 = st.sidebar.multiselect('구별_대분류', (my_df1.지번주소_대범주.unique()), default=(my_df1.지번주소_대범주.unique())) # default 값 -> 모든게 선택되어있게 설정
my_df1 = my_df1[my_df1.지번주소_대범주.isin(option02)]
option03 = st.sidebar.multiselect('구별_소분류', (my_df1.지번주소_소범주.unique()), default=(my_df1.지번주소_소범주.unique()))
my_df1 = my_df1[my_df1.지번주소_소범주.isin(option03)]

st.sidebar.header('🍎업태분류')
option04 = st.sidebar.multiselect('음식점 업태분류', (my_df1.업태구분명.unique()), default=(my_df1.업태구분명.unique())) # default 값 -> 모든게 선택되어있게 설정
my_df1 = my_df1[my_df1.업태구분명.isin(option04)]


# In[346]:


st.header('1. 인허가 및 폐업 현황 분석')

st.subheader('인허가')
time_frame = st.selectbox('연도별/월별',
                          ('인허가년도','인허가월')) #선택한 값이 저장됌
whole_values = my_df1.groupby(time_frame)[['개방자치단체코드']].count()
st.area_chart(whole_values, use_container_width = True)

st.subheader('폐업')
st.area_chart(전체폐업 , use_container_width = True)


# In[347]:


st.subheader('지역별 비교_인허가')

city_range = st.radio(label="범위선택", options=("구단위", "동단위"), index=0)

if city_range=='구단위':
    city_range='지번주소_대범주'
    small_region=False
else:
    city_range='지번주소_소범주'
    small_region = st.multiselect("동선택", (my_df1.지번주소_소범주.unique()), (my_df1.지번주소_소범주.unique()))

if small_region==False:
    city_values = my_df1
else:
    city_values = my_df1[my_df1['지번주소_소범주'].isin(small_region)]
    
city_values = pd.pivot_table(city_values, index=time_frame, columns=city_range, 
                             values='개방자치단체코드', aggfunc='count',fill_value=0)
city_values.index.name = None
city_values.columns = list(city_values.columns)

st.line_chart(city_values, use_container_width=True)

st.subheader('지역별 비교_폐업')
st.line_chart(구별폐업)


# In[348]:


st.subheader('2.음식점 지역별 분포')
my_df1 = my_df1.rename(columns={'경도':'lon','위도':'lat'})
jit = np.random.randn(len(my_df1), 2)
jit_ratio = 0.01
my_df1[['lat','lon']] = my_df1[['lat','lon']] + jit*jit_ratio
st.map(my_df1)

