import math
import torch

def evaluation(top_N, num_whole_items, max_length_test, train_positive, scores, test_positive):
    test_padding = []
    for i in range(len(train_positive)):
        user_negative_items_test = list(set(range(num_whole_items+1))-set(train_positive[i]))
        user_negative_items_test += [num_whole_items]*(max_length_test - len(user_negative_items_test))
        test_padding.append(user_negative_items_test)
    negative_scores = torch.gather(input=torch.FloatTensor(scores), dim=1, index=torch.LongTensor(test_padding))
    topk_indices = torch.gather(input=torch.LongTensor(test_padding), dim=1, index=torch.topk(input=negative_scores, k=top_N, dim=1, largest=True, sorted=True)[1]).tolist()
    count_hr_200 = 0
    count_ndcg_200 = 0

    count_hr_10 = 0
    count_ndcg_10 = 0

    count_hr_50 = 0
    count_ndcg_50 = 0

    count_hr_100 = 0
    count_ndcg_100 = 0

    for user_indices, user_positive_id in zip(topk_indices, test_positive):
        if user_positive_id in user_indices:
            count_hr_200 += 1
            idx_200 = user_indices.index(user_positive_id)+1
            count_ndcg_200 += math.log(2) / math.log(1 + idx_200)
            if idx_200 <= 10:
                count_hr_10 += 1
                idx_10 = idx_200
                count_ndcg_10 += math.log(2) / math.log(1 + idx_10)
            if idx_200 <= 50:
                count_hr_50 += 1
                idx_50 = idx_200
                count_ndcg_50 += math.log(2) / math.log(1 + idx_50)
            if idx_200 <= 100:
                count_hr_100 += 1
                idx_100 = idx_200
                count_ndcg_100 += math.log(2) / math.log(1 + idx_100)    
    return count_hr_10, count_ndcg_10, count_hr_50, count_ndcg_50, count_hr_100, count_ndcg_100, count_hr_200, count_ndcg_200