import torch
import torch.nn as nn

class AdaIN(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # self.hidden_dim = hidden_dim
        # self.EnDe1 = nn.Sequential(
        #     nn.Conv2d(hidden_dim, hidden_dim//2, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(hidden_dim // 2, hidden_dim // 8, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(hidden_dim // 8, hidden_dim // 2, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=1)
        # ).to('cuda')
        #
        # self.EnDe2 = nn.Sequential(
        #     nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(hidden_dim // 2, hidden_dim // 8, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(hidden_dim // 8, hidden_dim // 2, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=1)
        # ).to('cuda')

    def mu(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the average across
        it's spatial dimensions as (h,w) tensor [See eq. 5 of paper]"""
        return torch.sum(x,(2,3))/(x.shape[2]*x.shape[3])

    def sigma(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the standard deviation
        across it's spatial dimensions as (h,w) tensor [See eq. 6 of paper] Note
        the permutations are required for broadcasting"""
        return torch.sqrt((torch.sum((x.permute([2,3,0,1])-self.mu(x)).permute([2,3,0,1])**2,(2,3))+0.000000023)/(x.shape[2]*x.shape[3]))

    def forward(self, x, y):
        """ Takes a content embeding x and a style embeding y and changes
        transforms the mean and standard deviation of the content embedding to
        that of the style. [See eq. 8 of paper] Note the permutations are
        required for broadcasting"""
        # sigma_y_t = self.EnDe1(self.sigma(y).view(-1, self.hidden_dim, 1, 1)).squeeze()
        # mu_y_t = self.EnDe2(self.mu(y).view(-1, self.hidden_dim, 1, 1)).squeeze()
        return (self.sigma(y)*((x.permute([2,3,0,1])-self.mu(x))/self.sigma(x)) + self.mu(y)).permute([2,3,0,1])
        # return (sigma_y_t * ((x.permute([2, 3, 0, 1]) - self.mu(x)) / self.sigma(x)) + mu_y_t).permute([2, 3, 0, 1])