import torch
import numpy as np


class HawkesModel:

    def __init__(self, excitation, verbose=False, device='cpu'):
        """
        Initialize the model

        Arguments:
        ----------
        prior : Prior
            Prior object                  #先验？
        excitation: excitation
            Excitation object             #激励核对象
        """
        self.excitation = excitation
        self.n_jumps = None
        self.dim = None
        self.n_params = None
        self.n_var_params = None
        self._fitted = False
        self.verbose = verbose
        self.device = 'cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu'

    def set_data(self, events):
        """
        Set the data for the model
        """
        assert isinstance(events[0], torch.Tensor)    #event[i]是tensor。tensor的每一维是一个维度。
        # Set various util attributes
        self.dim = len(events)                        #维度。
        self.n_params = self.dim * (self.dim + 1)     #参数的数量，m*(m+1),包括w_ij和u_i。
        self.n_var_params = 2 * self.n_params         #变分变量数量,2*n_params?
        self.n_jumps = sum(map(len, events))          #所有事件数
        self.end_time = max([max(num) for num in events if len(num) > 0])    #结束时间
        self.events = events
        #if not self._fitted:
        self._init_cache()                        #执行_init_cache()
        #self._fitted = True

    def _init_cache(self):
        """
        caching the required computations
        #缓存需求的计算？

        cache[i][j,0,k]: float
            sum_{t^j < t^i_k} phi(t^i_k - t^j)
            This is used in k^th timestamp of node i, i.e., lambda_i(t^i_k)
        cache_integral: float
            used in the integral of intensity
        """
        self._cache = [torch.zeros(
            (self.dim, self.excitation.M, len(events_i)), dtype=torch.float64, device=self.device)
            for events_i in self.events]      #每一维，有参数。dim*M*事件数
        for i in range(self.dim):
            for j in range(self.dim):
                if self.verbose:              #输出初始化进度
                    print(f"\rInitialize cache {i*self.dim+j+1}/{self.dim**2}     ", end='')
                id_end = np.searchsorted(            #searchsorted（a，v）函数是判断v在a中哪两个a[n-1],a[n]之间，并返回n。
                                                    #searchsorted找出v中某个元素放在a中哪个位置上才能保持原有的排列顺序。
                    self.events[j].cpu().numpy(),
                    self.events[i].cpu().numpy())
                id_start = np.searchsorted(
                    self.events[j].cpu().numpy(),
                    self.events[i].cpu().numpy() - self.excitation.cut_off)  #cut_off是时间序列截断（事件序列考虑多长一段，可以认为是最大时间）。
                for k, time_i in enumerate(self.events[i]):
                    t_ij = time_i - self.events[j][id_start[k]:id_end[k]]    #如果cut_off是0的话，那么events[j][id_start[k]:id_end[k]]为空。
                    kappas = self.excitation.call(t_ij).sum(-1)              #如果是高斯混合模型的话，要加上tensor所有的值。其它情况就一个值。
                    self._cache[i][j, :, k] = kappas
        if self.verbose:
            print()

        self._cache_integral = torch.zeros((self.dim, self.excitation.M),
                                           dtype=torch.float64, device=self.device)
        for j in range(self.dim):
            t_diff = self.end_time - self.events[j]    #T-t
            integ_excit = self.excitation.callIntegral(t_diff).sum(-1)  # (M)
            self._cache_integral[j, :] = integ_excit

    #def log_likelihood(self, mu, W, epsilon_noise):
    def log_likelihood(self, mu, W):
        """
        Log likelihood of Hawkes Process for the given parameters mu and W
        #log likelihood

        Arguments:
        ----------
        mu : torch.Tensor
            (dim x 1)
            Base intensities
        W : torch.Tensor
            (dim x dim x M) --> M is for the number of different excitation functions
            The weight matrix.
        """
        log_like = 0
        for i in range(self.dim):
            # W[i] (dim x M)
            # _cache[i] (dim x M X len(events[i]))
            #print(self._cache)
            #print('/n')
            intens = torch.log(mu[i] + (W[i].unsqueeze(2) * self._cache[i]).sum(0).sum(0))
            #intens = torch.log(mu[i] + (W[i].unsqueeze(2) * self._cache[i]).sum(0).sum(0) - epsilon_noise)
            log_like += intens.sum()
            #print('log_like is:',log_like)
        log_like -= self._integral_intensity(mu, W)
        #log_like += self.dim*self.end_time*epsilon_noise
        return log_like

    def _integral_intensity(self, mu, W):
        """
        Integral of intensity function
        #强度函数积分

        Argument:
        ---------
        node_i: int
            Node id
        """
        integ_ints = (W * self._cache_integral.unsqueeze(0)).sum(1).sum(1)
        integ_ints += self.end_time * mu
        return integ_ints.sum()