import math
import torch
import torch.nn.functional as F
class Model(torch.nn.Module):
    def __init__(self, num_users, num_items, dim_embedding):
        super(Model,self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.dim_embedding = dim_embedding

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.dim_embedding)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.dim_embedding)
        self.scale_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=3)
        self.scale_item_1 = torch.nn.Parameter(torch.ones(1, self.num_items))
        self.scale_item_2 = torch.nn.Parameter(torch.ones(1, self.num_items)*5)
        self.scale_item_3 = torch.nn.Parameter(torch.ones(1, self.num_items)*5)
        self.mask = torch.ones(self.scale_item_1.shape).cuda()
        self.mask[:,-1] =0
        
        torch.nn.init.normal_(self.embedding_user.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.embedding_item.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.scale_user.weight, mean=1, std=0)
        
     
        self.weight = torch.nn.Parameter(torch.FloatTensor(self.dim_embedding, 1))
        torch.nn.init.uniform_(self.weight, a=-math.sqrt(1.0/self.dim_embedding), b=math.sqrt(1.0/self.dim_embedding))


    def forward(self, batch_users, whole_items, dropout_ration):
        self.batch_user_scale = self.scale_user(batch_users)
        batch_user_embeddings = torch.nn.functional.normalize(torch.nn.functional.dropout(self.embedding_user(batch_users), p=dropout_ration, training=True), dim=-1)
        whole_item_embeddings = self.embedding_item(whole_items)

        self.likelihood= torch.nn.functional.relu(torch.mm(batch_user_embeddings*self.weight.T, whole_item_embeddings.T))
        self.likelihood[:, -1] =0
    
    def compute_positive_loss(self, batch_positive_items_view, batch_positive_items_cart, batch_positive_items_buy):

        self.likelihood_positive_view = torch.gather(input=self.likelihood, dim=1, index=batch_positive_items_view)
        self.likelihood_positive_cart = torch.gather(input=self.likelihood, dim=1, index=batch_positive_items_cart)
        self.likelihood_positive_buy = torch.gather(input=self.likelihood, dim=1, index=batch_positive_items_buy)

        self.criterion_positive_view = torch.gather(input=self.batch_user_scale[:,0].unsqueeze(1)* self.scale_item_1*self.mask, dim = 1, index=batch_positive_items_view)
        self.criterion_positive_cart = torch.gather(input=self.batch_user_scale[:,1].unsqueeze(1)* self.scale_item_2*self.mask, dim = 1, index=batch_positive_items_cart)
        self.criterion_positive_buy = torch.gather(input=self.batch_user_scale[:,2].unsqueeze(1)* self.scale_item_3*self.mask, dim = 1, index=batch_positive_items_buy)

        self.loss_positive_view = torch.sum(torch.pow(torch.nn.functional.relu(self.criterion_positive_view  - self.likelihood_positive_view), 2))
        self.loss_positive_cart = torch.sum(torch.pow(torch.nn.functional.relu(self.criterion_positive_cart - self.likelihood_positive_cart), 2))
        self.loss_positive_buy = torch.sum(torch.pow(torch.nn.functional.relu(self.criterion_positive_buy - self.likelihood_positive_buy), 2))

    def compute_all_loss(self, weight_negative, lambdas, alpha):
        
        loss_negative_view = torch.sum(torch.pow(torch.nn.functional.relu(self.likelihood - alpha*self.batch_user_scale[:,0].unsqueeze(1)* self.scale_item_1*self.mask), 2)) - torch.sum(torch.pow(torch.nn.functional.relu(self.likelihood_positive_view - alpha* self.criterion_positive_view), 2))
        loss_negative_cart = torch.sum(torch.pow(torch.nn.functional.relu(self.likelihood - alpha*self.batch_user_scale[:,1].unsqueeze(1)* self.scale_item_2*self.mask), 2)) - torch.sum(torch.pow(torch.nn.functional.relu(self.likelihood_positive_cart - alpha* self.criterion_positive_cart), 2))
        loss_negative_buy =  torch.sum(torch.pow(torch.nn.functional.relu(self.likelihood - alpha*self.batch_user_scale[:,2].unsqueeze(1)* self.scale_item_3*self.mask), 2)) - torch.sum(torch.pow(torch.nn.functional.relu(self.likelihood_positive_buy - alpha* self.criterion_positive_buy), 2))

        loss_view = self.loss_positive_view + weight_negative * loss_negative_view
        loss_cart = self.loss_positive_cart + weight_negative * loss_negative_cart
        loss_buy = self.loss_positive_buy + weight_negative * loss_negative_buy

        self.loss = lambdas['view'] * loss_view + lambdas['cart'] * loss_cart + lambdas['buy'] * loss_buy
        return self.loss


    def predict(self, batch_users, whole_items):
        self.batch_user_scale = self.scale_user(batch_users)
        batch_user_embeddings = torch.nn.functional.normalize(self.embedding_user(batch_users), dim=-1)
        whole_item_embeddings = self.embedding_item(whole_items)

        self.likelihood= F.relu(torch.mm(batch_user_embeddings*self.weight.T, whole_item_embeddings.T))
        self.likelihood[:, -1] =0
        return self.likelihood/(F.relu(self.scale_item_3)+1e-4)
