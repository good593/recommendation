import numpy as np
import torch

__all__ = ['metrics']

# hit rate는 groud truth가 예측한 아이템 순위 k 안에 들어가는 비율을 나타낸 것
def __hit(gt_item, pred_items):
  _result = 0
  if gt_item in pred_items:
    _result = 1
    
  return _result

def __ndcg(gt_item, pred_items):
  _result = 0
  if gt_item in pred_items:
    index = pred_items.index(gt_item)
    _result = np.reciprocal(np.log2(index + 2))
    
  return _result

def metrics(model, test_loader, top_k):
  HR, NDCG = [], [] 
  for user, item, _ in test_loader:
    # user = user.cuda()
    # item = item.cuda()
    predictions = model(user, item)
    # 가장 높은 top_k개 선택
    _, indices = torch.topk(predictions, top_k)
    # 해당 상품 index 선택 
    recommends = torch.take(item, indices).cpu().numpy().tolist()
    # 정답값 선택
    gt_item = item[0].item()
    HR.append(__hit(gt_item, recommends))
    NDCG.append(__ndcg(gt_item, recommends))

  return np.mean(HR), np.mean(NDCG)
