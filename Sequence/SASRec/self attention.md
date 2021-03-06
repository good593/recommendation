# [self attention](https://ratsgo.github.io/nlpbook/docs/language_model/tr_self_attention/)
> 트랜스포머(transformer)의 핵심 구성요소는 self attention입니다.  
## 1. 모델 입력과 출력 
- 입력층
  - 인코더 입력은 소스 시퀀스의 입력 임베딩(input embedding)에 위치 정보(positional encoding)을 더해서 만듭니다. 즉, 소스 언어의 토큰 시퀀스(입력 임베딩)를 이에 대응하는 벡터 시퀀스(+위치 정보)로 변환해서 인코더 입력을 만듭니다.  
- 출력층
  - 디코더 마지막 블록의 출력 벡터 시퀀습입니다. 출력층의 출력은 타깃 언어의 어휘 수 만큼의 차원을 갖는 확률 벡터가 됩니다. 
## 2. 셀프 어텐션 내부 동작
> 셀프 어텐션은 쿼리(query), 키(key), 벨류(value) 3개 요소 사이의 문맥적 관계성을 추출하는 과정입니다.    
> $Attention(Q,K,V) = softmax({QK^T \over \sqrt{d_K} })V$  
> 위 수식을 말로 풀면 이렇습니다. 쿼리와 키를 행렬곱한 뒤 해당 행렬의 모든 요소값을 키 차원수의 제곱근 값으로 나눠주고, 이 행렬을 행(row) 단위로 softmax를 취해 스코어 행렬을 만들어줍니다. 이 스코어 행렬에 벨류를 행렬곱해 줘서 셀프 어텐션 계산을 마칩니다.  

## 3. 멀티 헤드 어텐션
> Multi-Head Attention은 self attention을 여러 번 수행한 걸 가리킵니다. 여러 헤드가 독자적으로 셀프 어텐션을 계산한다는 이야기입니다. 비유하자면 같은 문서(입력)를 두고 독자(헤드) 여러 명이 함께 읽는 구조라 할 수 있겠습니다.

## 4. 인코더에서 수행하는 셀프 어텐션
- 트랜스포머 인코더에서 수행하는 계산 과정을 셀프 어텐션을 중심으로 살펴보겠습니다. 트랜스포머 인코더 블록의 입력은 이전 블록의 단어 벡터 시퀀스, 출력은 이번 블록 수행 결과로 도출된 단어 벡터 시퀀스입니다.     
- 인코더에서 수행되는 셀프 어텐션은 쿼리, 키, 밸류가 모두 소스 시퀀스와 관련된 정보입니다. 트랜스포머의 학습 과제가 한국어에서 영어로 번역하는 테스트라면, 인코더의 쿼리, 키, 밸류는 모두 한국어가 된다는 이야기입니다.  


## 5. 디코더에서 수행하는 셀프 어텐션
- 디코더 입력은 인코더 마지막 블록에서 나온 소스 단어 벡터 시퀀스, 이전 디코더 블록의 수행 결과로 도출된 타깃 단어 벡터 시퀀스입니다.  
- 디코더에서 수행되는 셀프 어텐션 순서
  - Masked Multi-Head Attention: 이 모듈에서는 타깃 언어의 단어 벡터 시퀀스를 계산 대상으로 합니다. 한국어를 영어로 번역하는 테스트를 수행하는 트랜스포머 모델이라면 여기서 계산되는 대상은 영어 단어 시퀀스가 됩니다.    
  - Multi-Head Attention: 인코더와 디코더 쪽 정보를 모두 활용합니다. 인코더에서 넘어온 정보는 소스 언어의 문장의 단어 벡터 시퀀스입니다. 디코더 정보는 타깃 언어 문장의 단어 벡터 시퀀스입니다. 전자를 키, 후자를 쿼리로 삼아 셀프 어텐션 계산을 수행합니다.  



