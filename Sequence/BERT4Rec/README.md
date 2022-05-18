# [BERT4REC](https://www.youtube.com/watch?v=PKYVHGrSO2U)
- https://greeksharifa.github.io/machine_learning/2021/12/12/Bert4Rec/
- https://22-22.tistory.com/78
- https://hyunlee103.tistory.com/115
### github
- https://github.com/FeiSun/BERT4Rec
## Background
> RNN을 필두로 한 left-to-right unidirectional model은 user behavior sequence를 파악하기에 충분하지 않습니다. 왜냐하면 user의 historical interaction에서 일어난 item 선택 과정에 대해 살펴보면, 여러 이유로 인해 꼭 그 순서 자체가 중요하다고 말할 수 없는 경우가 자주 발생하기 때문입니다. 예를 들어 어떤 user가 토너와 화장솜을 사고 싶다고 할 때 토너를 먼저 구매할 수도 있고, 화장솜을 먼저 구매할 수도 있습니다. 사실 어떤 것이 먼저 오냐는 관점에 따라 크게 중요하지 않은 사실이 될 가능성이 높습니다.  
  
> 따라서 논문에서는 sequence representations learning을 위해 두 방향 모두에서 context를 통합해야 한다고 이야기합니다. 