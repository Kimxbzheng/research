import torch
import torch.utils.data
import visdom
import collections
import numpy as np
from collections import defaultdict

vis = visdom.Visdom(port=12345)

train_dataset = torch.utils.data.TensorDataset(torch.randn(5000, 4, 240), torch.randn(5000, 4, 30))
test_dataset  = torch.utils.data.TensorDataset(torch.randn(500, 4, 240), torch.randn(500, 4, 30))

train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=32)
test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=32)

def cycle(iterable):
	while True:
		for x in iterable:
			yield x

train_iterator = iter(cycle(train_loader))
test_iterator = iter(cycle(test_loader))

def Block(f_in, f_out):
	layers = []
	layers.append(nn.Conv1d(f_in, f_out, 4, 2 ,1))
	layers.append(nn.BatchNorm1d(f_out))
	layers.append(nn.ReLU())
	return nn.Sequential(*layers)

def FlatBlock(f_in):
	layers = []
	layers.append(nn.Conv1d(f_in, f_in, 3, 1 ,1))
	layers.append(nn.BatchNorm1d(f_out))
	layers.append(nn.ReLU())
	return nn.Sequential(*layers)

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		f = []
		f.append(Block(4, 32))
		f.append(Block(32,64))

		# f.append(FlatBlock(64)) # if underfitting and increasing initial channels doesn't help
		# f.append(FlatBlock(64))
		# f.append(FlatBlock(64))
		# f.append(FlatBlock(64))

		f.append(Block(64,4))
		self.f = nn.Sequential(*f)

	def forward(self,x):
		return self.f(x)

net = Net()
opt = torch.optim.Adam(net.parameters(), weight_decay=0.001, lr=0.001) # tune weight decay
print(f'> Number of network parameters {len(torch.nn.utils.parameters_to_vector(net.parameters()))}')

iter_per_epoch = 100
epoch = 0

vis.line(X=np.array([0]), Y=np.array([[np.nan, np.nan]]), win='loss')

while True:

	stats = defaultdict(list)

	for i in range(iter_per_epoch):

		x,t = next(train_iterator)
		p = net(x)

		loss = ((t-p)**2).mean()
		# loss.mean(dim=2) # remove previous mean, multiply column then take mean to weight per channel
		opt.zero_grad()
		loss.backward()
		opt.step()

		stats['loss'] = np.append(stats['loss'], loss.item())
		
	vis.line(X=np.array([epoch]), Y=np.array([[
        stats['loss'].mean(),
        stats['loss'].std()
    ]]), win='loss', opts=dict(title='loss',xlabel='epoch', ylabel='stats', ytype='log', legend=[
        'mean',
        'std'
    ]), update='append')


	epoch += 1