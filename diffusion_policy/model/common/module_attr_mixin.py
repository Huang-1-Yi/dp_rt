import torch.nn as nn

class ModuleAttrMixin(nn.Module):
    def __init__(self):
        super().__init__()
        """
        修改ModuleAttrMixin中的虚拟参数设置不会影响ObsEncoder的功能和性能，但是能减小占用和加速训练
        反向传播时，跳过虚拟参数梯度（节省计算），参数统计更准确。
        """
        # self._dummy_variable = nn.Parameter()
        self._dummy_variable = nn.Parameter(requires_grad=False)
        

    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
