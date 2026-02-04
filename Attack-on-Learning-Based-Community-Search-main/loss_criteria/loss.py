import torch
from torch.nn import BCELoss
from torch.nn import functional as F
from torch.nn.modules.module import Module


class WeightBCEWithLogitsLoss(Module):
	"""
	BCEwithLogitsLoss with element-wise weight
	"""
	def __init__(self):
		super(WeightBCEWithLogitsLoss, self).__init__()
		self.bce = BCELoss(reduction="none")

	def forward(self, inputs, target, weights):
		loss = self.bce(inputs, target)
		if weights is not None:
			loss = torch.mul(loss, weights)
		loss = torch.sum(loss)
		return loss


class WeightedBinaryCrossEntropy(Module):
    def __init__(self, weight=None):
        super(WeightedBinaryCrossEntropy, self).__init__()
        if weight is not None and not isinstance(weight, torch.Tensor):
            weight = torch.tensor(weight, dtype=torch.float32)
        self.weight = weight

    def forward(self, predictions, targets):
        epsilon = 1e-9

        predictions = torch.clamp(predictions, epsilon, 1 - epsilon)
        loss = self.weight * (targets * torch.log(predictions) + (1 - targets) * torch.log(1 - predictions))
        
        return torch.sum(loss)
    

class WeightedKLDivergence(Module):
    def __init__(self, weight=None):
        super(WeightedKLDivergence, self).__init__()
        if weight is not None and not isinstance(weight, torch.Tensor):
            weight = torch.tensor(weight, dtype=torch.float32)
        self.weight = weight

    def forward(self, predictions, targets):
        epsilon = 1e-9

        # 确保 predictions 和 targets 是概率分布
        predictions = torch.clamp(predictions, epsilon, 1 - epsilon)
        targets = torch.clamp(targets, epsilon, 1 - epsilon)

        # 计算 KL 散度
        loss = self.weight * F.kl_div(predictions.log(), targets, reduction='batchmean')
        
        return loss