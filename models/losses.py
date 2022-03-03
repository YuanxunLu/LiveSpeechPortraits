import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F


class GMMLogLoss(nn.Module):
    ''' compute the GMM loss between model output and the groundtruth data.
    Args:
        ncenter: numbers of gaussian distribution
        ndim: dimension of each gaussian distribution
        sigma_bias:
        sigma_min:  current we do not use it.
    '''
    def __init__(self, ncenter, ndim, sigma_min=0.03):
        super(GMMLogLoss,self).__init__()
        self.ncenter = ncenter
        self.ndim = ndim
        self.sigma_min = sigma_min       
    
    
    def forward(self, output, target):
        '''
        Args:
            output: [b, T, ncenter + ncenter * ndim * 2]:
                [:, :,  : ncenter] shows each gaussian probability 
                [:, :, ncenter : ncenter + ndim * ncenter] shows the average values of each dimension of each gaussian 
                [: ,:, ncenter + ndim * ncenter : ncenter + ndim * 2 * ncenter] show the negative log sigma of each dimension of each gaussian 
            target: [b, T, ndim], the ground truth target landmark data is shown here 
        To maximize the log-likelihood equals to minimize the negative log-likelihood. 
        NOTE: It is unstable to directly compute the log results of sigma, e.g. ln(-0.1) as we need to clip the sigma results 
        into positive. Hence here we predict the negative log sigma results to avoid numerical instablility, which mean:
            `` sigma = 1/exp(predict), predict = -ln(sigma) ``
        Also, it will be just the 'B' term below! 
        Currently we only implement single gaussian distribution, hence the first values of pred are meaningless.
        For single gaussian distribution:
            L(mu, sigma) = -n/2 * ln(2pi * sigma^2) - 1 / (2 x sigma^2) * sum^n (x_i - mu)^2  (n for prediction times, n=1 for one frame, x_i for gt)
                         = -1/2 * ln(2pi) - 1/2 * ln(sigma^2) - 1/(2 x sigma^2) * (x - mu)^2
        == min -L(mu, sgima) = 0.5 x ln(2pi) + 0.5 x ln(sigma^2) + 1/(2 x sigma^2) * (x - mu)^2
                             = 0.5 x ln_2PI + ln(sigma) + 0.5 x (MU_DIFF/sigma)^2
                             = A - B + C
            In batch and Time sample, b and T are summed and averaged.
        '''
        b, T, _ = target.shape
        # read prediction paras
        mus = output[:, :, self.ncenter : (self.ncenter + self.ncenter * self.ndim)].view(b, T, self.ncenter, self.ndim)  # [b, T, ncenter, ndim]
        
        # apply min sigma
        neg_log_sigmas_out = output[:, :, (self.ncenter + self.ncenter * self.ndim):].view(b, T, self.ncenter, self.ndim)  # [b, T, ncenter, ndim]   
        inv_sigmas_min = torch.ones(neg_log_sigmas_out.size()).cuda() * (1. / self.sigma_min)
        inv_sigmas_min_log = torch.log(inv_sigmas_min)
        neg_log_sigmas = torch.min(neg_log_sigmas_out, inv_sigmas_min_log)       
        
        inv_sigmas = torch.exp(neg_log_sigmas)
        # replicate the target of ncenter to minus mu
        target_rep = target.unsqueeze(2).expand(b, T, self.ncenter, self.ndim)  # [b, T, ncenter, ndim]
        MU_DIFF = target_rep - mus  # [b, T, ncenter, ndim]
        # sigma process
        A = 0.5 * math.log(2 * math.pi)   # 0.9189385332046727
        B = neg_log_sigmas  # [b, T, ncenter, ndim]
        C = 0.5 * (MU_DIFF * inv_sigmas)**2  # [b, T, ncenter, ndim]
        negative_loglikelihood =  A - B + C  # [b, T, ncenter, ndim]
        
        return negative_loglikelihood.mean()


def Sample_GMM(gmm_params, ncenter, ndim, weight_smooth = 0.0, sigma_scale = 0.0):
    ''' Sample values from a given a GMM distribution.
    Args:
        gmm_params: [b, target_length, (2 * ndim + 1) * ncenter], including the 
        distribution weights, average and sigma
        ncenter: numbers of gaussian distribution
        ndim: dimension of each gaussian distribution 
        weight_smooth: float, smooth the gaussian distribution weights
        sigma_scale: float, adjust the gaussian scale, larger for sharper prediction,
            0 for zero sigma which always return average values
    Returns:
        current_sample: []
    '''
    # reshape as [b*T, (2 * ndim + 1) * ncenter]
    b, T, _ = gmm_params.shape
    gmm_params_cpu = gmm_params.cpu().view(-1, (2 * ndim + 1) * ncenter)
    # compute each distrubution probability
    prob = nn.functional.softmax(gmm_params_cpu[:, : ncenter] * (1 + weight_smooth), dim=1)
    # select the gaussian distribution according to their weights
    selected_idx = torch.multinomial(prob, num_samples=1, replacement=True)
    
    mu = gmm_params_cpu[:, ncenter : ncenter + ncenter * ndim]
    # please note that we use -logsigma as output, hence here we need to take the negative
    sigma = torch.exp(-gmm_params_cpu[:, ncenter + ncenter * ndim:]) * sigma_scale
#    print('sigma average:', sigma.mean())
    
    selected_sigma = torch.empty(b*T, ndim).float()
    selected_mu = torch.empty(b*T, ndim).float()
    current_sample = torch.randn(b*T, ndim).float()
#    current_sample = test_sample

    for i in range(b*T):
        idx = selected_idx[i, 0]
        selected_sigma[i, :] = sigma[i, idx * ndim:(idx + 1) * ndim]
        selected_mu[i, :] = mu[i, idx * ndim:(idx + 1) * ndim]

    # sample with sel sigma and sel mean
    current_sample = current_sample * selected_sigma + selected_mu
    # cur_sample = sel_mu
#    return  current_sample.unsqueeze(1).cuda()

    if torch.cuda.is_available():
        return  current_sample.reshape(b, T, -1).cuda()
    else:
        return  current_sample.reshape(b, T, -1)



class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor        
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None        
        gpu_id = input.get_device()
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).cuda(gpu_id).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).cuda(gpu_id).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)                
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)




class VGGLoss(nn.Module):
    def __init__(self, model=None):
        super(VGGLoss, self).__init__()
        if model is None:
            self.vgg = Vgg19()
        else:
            self.vgg = model

        self.vgg.cuda()
        # self.vgg.eval()
        self.criterion = nn.L1Loss()
        self.style_criterion = StyleLoss()
        self.weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        self.style_weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        # self.weights = [5.0, 1.0, 0.5, 0.4, 0.8]
        # self.style_weights = [10e4, 1000, 50, 15, 50]

    def forward(self, x, y, style=False):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        if style:
            # return both perceptual loss and style loss.
            style_loss = 0
            for i in range(len(x_vgg)):
                this_loss = (self.weights[i] *
                             self.criterion(x_vgg[i], y_vgg[i].detach()))
                this_style_loss = (self.style_weights[i] *
                                   self.style_criterion(x_vgg[i], y_vgg[i].detach()))
                loss += this_loss
                style_loss += this_style_loss
            return loss, style_loss

        for i in range(len(x_vgg)):
            this_loss = (self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach()))
            loss += this_loss
        return loss
    

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)
    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
    G = torch.mm(features, features.t())  # compute the gram product
    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()

    def forward(self, x, y):
        Gx = gram_matrix(x)
        Gy = gram_matrix(y)
        return F.mse_loss(Gx, Gy) * 30000000

        

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, input, target, mask):        
        mask = mask.expand(-1, input.size()[1], -1, -1)
        loss = self.criterion(input * mask, target * mask)
        return loss



from torchvision import models
class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out











