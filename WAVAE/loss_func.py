import torch
import torch.nn as nn
import torch.nn.functional as F


def vae_loss(pred, y_real, cur_cnt, padding, hyper_params):
    pred_out = F.log_softmax(pred, dim=-1)
    y_real = F.one_hot(y_real, num_classes=hyper_params['total_items'] + 1)
    padding = (1.0 - padding.float()).unsqueeze(2)

    likelihood = -1.0 * \
        torch.sum(pred_out * y_real * padding) / float(pred.shape[0])
    return likelihood

def G_vae(D,pred, y_real, cur_cnt, padding, hyper_params):
    pred_out=torch.randn_like(pred)
    fake=D(pred)
    PRIOR = -torch.mean(fake, dim=-1) * (1.0 - padding.float())
    PRIOR = torch.sum(PRIOR) / pred_out.shape[0] * hyper_params['D_weight']
    return PRIOR

def D_vae(D,pred, y_real, cur_cnt, padding, hyper_params):
    pred_out=torch.randn_like(pred)
    y_real = F.one_hot(y_real, num_classes=hyper_params['total_items'] + 1)
    fake=D(pred)
    real=D(y_real)
    PRIOR = -torch.mean(real, dim=-1) * (1.0 - padding.float())+torch.mean(fake, dim=-1) * (1.0 - padding.float())
    PRIOR = torch.sum(PRIOR) / pred_out.shape[0]
    return PRIOR


def kl_loss(adversary, x_real, z_inferred, padding, KL_WEIGHT):
    t_joint = adversary(x_real, z_inferred, padding)
    t_joint = torch.mean(t_joint, dim=-1)
    kl = torch.sum(t_joint) / float(x_real.shape[0]) * KL_WEIGHT
    return kl

def adversary_kl_loss(adversary_prior, x_real, z_inferred, padding):
    prior = torch.randn_like(z_inferred)
    y_r=adversary_prior(x_real, z_inferred, padding)
    y_f=adversary_prior(x_real,prior,padding)
    term_a = torch.log(torch.sigmoid(y_r) + 1e-9)
    term_b = torch.log(1.0-torch.sigmoid(y_f) + 1e-9)
    PRIOR = -torch.mean(term_a+term_b , dim=-1) * (1.0 - padding.float())
    PRIOR = torch.sum(PRIOR) / x_real.shape[0]
    return PRIOR


class MetricShower():
    def __init__(self):
        self.metrics = {}
        self.metrics_cnt = {}

    def store(self, metrics: dict):
        for (k, v) in metrics.items():
            if self.metrics.get(k) is None:
                self.metrics[k] = 0.0
                self.metrics_cnt[k] = 0.0
            self.metrics[k] += v
            self.metrics_cnt[k] += 1

    def get(self, name):
        result = self.metrics[name] / self.metrics_cnt[name]

        return result

    def clear(self):
        for (k, v) in self.metrics.items():
            self.metrics[k] = 0.0
            self.metrics_cnt[k] = 0
