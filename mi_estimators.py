import numpy as np
import math

import torch 
import torch.nn as nn

class CLUBForCategorical(nn.Module):

    def __init__(self, input_dim, label_num, hidden_size=None):
        super().__init__()


        if hidden_size is None:
            self.variational_net = nn.Linear(input_dim, label_num)
        else:
            self.variational_net = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, label_num)
            )

            
    def forward(self, inputs, labels):
        '''
        inputs : shape [batch_size, x_dim]
        labels : shape [batch_size]
        '''
        logits = self.variational_net(inputs)  #[n_sample, y_dim]
        
        # log of conditional probability of positive sample pairs
        #positive = - nn.functional.cross_entropy(logits, labels, reduction='none')
        
        sample_size, label_num = logits.shape
        
        logits_extend = logits.unsqueeze(1).repeat(1, sample_size, 1)    # shape [sample_size, sample_size, dim]
        labels_extend = labels.unsqueeze(0).repeat(sample_size, 1)    # shape [sample_size, sample_size]


        # log of conditional probability of negative sample pairs
        log_mat = - nn.functional.cross_entropy(
            logits_extend.reshape(-1, label_num),
            labels_extend.reshape(-1, ),
            reduction='none'
        )

        log_mat = log_mat.reshape(sample_size, sample_size)
        #print(log_mat)

        positive = torch.diag(log_mat).mean()

        negative = log_mat.mean()
        
        return positive - negative


    def loglikeli(self, inputs, labels):
        logits = self.variational_net(inputs)
        return - nn.functional.cross_entropy(logits, labels)
    
    def learning_loss(self, inputs, labels):
        return - self.loglikeli(inputs, labels)


if __name__ == '__main__':
    input_dim = 10
    sample_size= 100
    
    mi_estimator = CLUBForCategorical(input_dim=input_dim, label_num=2)

    samples1 = torch.randn(sample_size//2, input_dim)

    samples2 = torch.randn(sample_size//2, input_dim) + 1.

    sample = torch.cat([samples1, samples2], dim=0)

    label = torch.cat([torch.zeros(sample_size//2), torch.ones(sample_size//2)]).long()

    print(mi_estimator.learning_loss(sample, label))

    print(mi_estimator(sample, label))
          
