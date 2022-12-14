from evaluator import Evaluator
import json
import torch
import torch.utils.data as Data
import model
import dataset_load
import numpy as np
import os
import logging
import traceback
import loss_func
import random
import time
import argparse
from tensorboardX import SummaryWriter

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

hyper_params = {
    'dataset_path': 'ml-1m',
    'kl_weight': 0.05,
    'D_weight': 1.0,
    'epochs': 500,
    'evaluate_users': None,
    'item_embed_size': 128,
    'rnn_size': 100,
    'hidden_size': 100,
    'latent_size': 64,
    'timesteps': 5,
    'test_prop': 0.2,
    'batch_size': 64,
    'anneal': False,
    'time_split': True,
    'model_func': 'fc_cnn',
    'add_eps': True,
    'device': 'cpu',
    'check_freq': 4,
    'Ks': [1, 5, 10, 20, 50, 100],
    'lr_primal': 1e-3,
    'lr_dual': 3e-4,
    'lr_prior': 5e-4,
    'l2_regular': 1e-2,
    'l2_adver': 1e-1,
    'total_step': 2000000,
    'short_prop':1,
    'seq_len1': 5,
    'seq_num': 10,
    'layer_num': 2,
    'dim': 64
}
dataset_info = json.load(open('./dataset_info.json', 'rt')
                         )[hyper_params['dataset_path']]
hyper_params['total_users'] = dataset_info[0]
hyper_params['total_items'] = dataset_info[1]
hyper_params['seq_len'] = dataset_info[2]

info_str = 'dataset:' + hyper_params['dataset_path'] + ' lr1:' + str(hyper_params["lr_primal"]) + ' lr2:' + str(
    hyper_params['lr_dual']) + ' kl:' + str(hyper_params['kl_weight']) + ' batch:' + str(hyper_params['batch_size']) + ' model_func:' + hyper_params['model_func']
path_str = f'{hyper_params["model_func"]}_{hyper_params["dataset_path"]}_kl_{hyper_params["kl_weight"]}_dropout_0.5_addeps_{hyper_params["add_eps"]}'

parser = argparse.ArgumentParser(prog='train')
parser.add_argument("-m", "--msg", default="no description")
args = parser.parse_args()
train_msg = args.msg

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(1111)

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
local_time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
handler = logging.FileHandler(
    "model_log/log_" + local_time_str + '_' + train_msg.replace(' ', '_') + ".txt")

handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.info(train_msg)
logger.info(info_str)
logger.info('Using CUDA:' + os.environ['CUDA_VISIBLE_DEVICES'])

dataset_load.load_data(hyper_params)
user_dataset = dataset_load.generate_train_data(hyper_params)

user_dataloader = Data.DataLoader(
    user_dataset, batch_size=hyper_params['batch_size'], shuffle=True)

test_dataset = dataset_load.generate_test_data(hyper_params)
test_dataloader = Data.DataLoader(
    test_dataset, batch_size=hyper_params['batch_size'], shuffle=True)

val_dataset = dataset_load.generate_validate_data(hyper_params)
val_dataloader = Data.DataLoader(
    val_dataset, batch_size=hyper_params['batch_size'], shuffle=True)

print('Building net...')
logger.info('Building net...')

net = model.Model(hyper_params).to(hyper_params['device'])
adversary = model.Adversary(hyper_params).to(hyper_params['device'])
D=model.D(hyper_params)
print(net)
print('Net build finished.')
logger.info('Net build finished.')

optimizer_primal = torch.optim.AdamW([{
    'params': net.parameters(),
    'lr': hyper_params['lr_primal'],
    'weight_decay': hyper_params['l2_regular']}
])
optimizer_dual = torch.optim.SGD([{
    'params': net.encoder.parameters(),
    'lr': hyper_params['lr_dual'],
    'weight_decay': hyper_params['l2_adver']
},
{
    'params':net.decoder.parameters(),
    'lr': hyper_params['lr_dual'],
    'weight_decay': hyper_params['l2_adver']
}
])
optimizer_prior = torch.optim.SGD([{
    'params': adversary.parameters(),
    'lr': hyper_params['lr_prior'],
    'weight_decay': hyper_params['l2_adver']}
])
optimizer_D=torch.optim.SGD([{
    'params': D.parameters(),
    'lr':hyper_params['lr_prior'],
    'weight_decay': hyper_params['l2_adver']}
])


print('User datasets loaded and saved.')
logger.info('User datasets loaded and saved.')

evaluator = Evaluator(hyper_params=hyper_params, logger=logger)


def train():
    writer = SummaryWriter(f'./runs/{path_str}')
    print('Start training...')
    logger.info('Start training...')

    global_step = 0
    mebank = loss_func.MetricShower()


    for epoch in range(hyper_params['epochs']):
        net.train()
        for batchx, batchy, padding, user_id, cur_cnt in user_dataloader:
            batchx = batchx.to(hyper_params['device'])
            batchy = batchy.to(hyper_params['device'])
            padding = padding.to(hyper_params['device'])
            user_id = user_id.to(hyper_params['device'])
            cur_cnt = cur_cnt.to(hyper_params['device'])
            optimizer_primal.zero_grad()
            optimizer_dual.zero_grad()
            optimizer_D.zero_grad()

            pred, x_real, z_inferred, out_embed = net(batchx)
            # --------------------------VAE---------------------------
            multi_loss = loss_func.vae_loss(
                pred, batchy, cur_cnt, padding, hyper_params)
            if hyper_params['anneal']:
                anneal = global_step / \
                    hyper_params['total_step'] * hyper_params['kl_weight']
            else:
                anneal = hyper_params['kl_weight']

            kl_loss = loss_func.kl_loss(
                adversary, x_real, z_inferred, padding, KL_WEIGHT=anneal)
            G_loss = loss_func.G_vae(D,pred, batchy, cur_cnt, padding, hyper_params)
            D_loss = loss_func.D_vae(D,pred, batchy, cur_cnt, padding,hyper_params)
            loss = multi_loss + kl_loss +G_loss
            loss.backward(retain_graph=True)
            D_loss.backward()

            optimizer_primal.step()
            optimizer_dual.step()
            optimizer_D.step()
            for p in D.parameters():
                p.data.clamp_(-0.01, 0.01)

            optimizer_prior.zero_grad()
            adver_kl_loss = loss_func.adversary_kl_loss(
                adversary, x_real.detach(), z_inferred.detach(), padding)

            adver_kl_loss.backward()
            optimizer_prior.step()
            # for q in adversary.parameters():
            #   q.data.clamp_(-0.01, 0.01)
            mebank.store({'vae': multi_loss.item(), 'kl': kl_loss.item(), 'prior': adver_kl_loss.item(),'Dloss':D_loss.item()})

            global_step += 1

        print(
            f'EPOCH:({epoch}/{hyper_params["epochs"]}),STEP:{global_step}/{hyper_params["total_step"]},vae:{mebank.get("vae")},kl:{mebank.get("kl")},prior:{mebank.get("prior")},Dloss:{mebank.get("Dloss")}')
        logger.info(
            f'EPOCH:({epoch}/{hyper_params["epochs"]}),STEP:{global_step}/{hyper_params["total_step"]},vae:{mebank.get("vae")},kl:{mebank.get("kl")},prior:{mebank.get("prior")},Dloss:{mebank.get("Dloss")}')

        writer.add_scalar('loss', mebank.get("vae"), global_step=epoch)
        writer.flush()

        mebank.clear()
        if epoch % hyper_params['check_freq'] == 0:
            hr, _, _ = evaluator.evaluate(net, adversary, dataloader=val_dataloader,
                                          validate=True, evaluate_users=hyper_params['evaluate_users'])
            writer.add_scalar('hr10', hr[2], global_step=epoch)
            writer.flush()
            net.train()
            adversary.train()

        if global_step >= hyper_params['total_step']:
            break

    evaluator.evaluate(net, adversary, dataloader=test_dataloader,
                       validate=False)
    writer.close()

if __name__ == '__main__':
    try:
        train()
        logger.info('Finished.')
    except Exception as err:
        err_info = traceback.format_exc()
        print(err_info)
        logger.info(err_info)
        logger.info('Error.')
