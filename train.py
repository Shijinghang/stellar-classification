import argparse
import time
from math import cos, pi

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from utlis.model import SCNet
import yaml
from sklearn.metrics import classification_report

from utlis.datasets import *
from utlis.metric import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def adjust_learning_rate(optimizer, current_epoch, max_epoch, lr0=0.0, lrf=0.01, warmup_epochs=0.0):
    if current_epoch < warmup_epochs:
        lr = lr0 * current_epoch / warmup_epochs
    else:
        lr = lrf + (lr0 - lrf) * (
                1 + cos(pi * (current_epoch - warmup_epochs) / (max_epoch - warmup_epochs))) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def build_optimizer(model, name='auto', lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
    if name == 'auto':
        nc = getattr(model, 'nc', 10)  # number of classes
        lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
        name, lr, momentum = ('SGD', 0.01, 0.9) if iterations > 10000 else ('AdamW', lr_fit, 0.9)

    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            fullname = f'{module_name}.{param_name}' if module_name else param_name
            if 'bias' in fullname:  # bias (no decay)
                g[2].append(param)
            elif isinstance(module, bn):  # weight (no decay)
                g[1].append(param)
            else:  # weight (with decay)
                g[0].append(param)

    if name in ('Adam', 'Adamax', 'AdamW', 'NAdam', 'RAdam'):
        optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    elif name == 'RMSProp':
        optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
    elif name == 'SGD':
        optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    else:
        raise NotImplementedError(
            f"Optimizer '{name}' not found in list of available optimizers "
            f'[Adam, AdamW, NAdam, RAdam, RMSProp, SGD, auto].'
            'To request support for addition optimizers please visit https://github.com/ultralytics/ultralytics.')

    optimizer.add_param_group({'params': g[0], 'weight_decay': decay})  # add g0 with weight_decay
    optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)
    return optimizer


def config_model(model, opt, device):
    if opt.pretrained:
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in torch.load(opt.weights).items() if
                           np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    if device.type == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model = model.cuda()

    return model


def fit_one_epoch(model, train_loader, val_loader, loss_func, current_epoch, max_epoch, opt, names, device):
    global best_acc
    start_time = time.time()

    # train
    model.train()
    train_pred, train_ture = [], []
    train_loss = 0
    with tqdm(total=len(train_loader.batch_sampler), desc=f'Epoch {current_epoch + 1}/{max_epoch}', postfix=dict,
              mininterval=0.3) as pbar:

        for iteration, batch in enumerate(train_loader):
            datas, labels = batch[0].to(device), batch[1].to(device)

            outputs = model(datas)
            loss = loss_func(outputs, labels)
            train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_pred += outputs.argmax(dim=1).cpu()
            train_ture += labels.cpu()

            waste_time = time.time() - start_time
            pbar.set_postfix(**{'train_loss': train_loss.item() / (iteration + 1),
                                'lr': get_lr(optimizer),
                                'step/s': waste_time})
            pbar.update(1)
            start_time = time.time()

    # Save metrics
    train_loss = train_loss.item() / (iteration + 1)
    train_report = classification_report(train_pred, train_ture, target_names=names, output_dict=True)
    save_metric(f'{opt.save_path}/train', train_report, train_loss, names)
    print_result(train_report)

    # Verify
    model.eval()
    val_pred, val_ture = [], []
    val_loss = 0
    with tqdm(total=len(val_loader.batch_sampler), desc=f'Epoch {current_epoch + 1}/{max_epoch}', postfix=dict,
              mininterval=0.3) as pbar:
        for iteration, batch in enumerate(val_loader):
            datas, labels = batch[0].to(device), batch[1].to(device)
            with torch.no_grad():
                datas = datas
                labels = labels.cuda()
                optimizer.zero_grad()
                outputs = model(datas)
                loss = loss_func(outputs, labels)
                val_loss += loss

                val_pred += outputs.argmax(dim=1).cpu()
                val_ture += labels.cpu()
            pbar.set_postfix(**{'val_loss': val_loss.item() / (iteration + 1)})
            pbar.update(1)

    # Calculate validation metrics and save
    val_loss = val_loss.item() / (iteration + 1)
    val_report = classification_report(val_pred, val_ture, target_names=names, output_dict=True)
    save_metric(f'{opt.save_path}/val', val_report, val_loss, names)
    print_result(val_report)

    # Save optimal model
    if val_report['accuracy'] > best_acc:
        best_acc = val_report['accuracy']
        torch.save(model.state_dict(), f'{opt.save_path}/best.pth')


def train(model, loss_func, train_loader, val_loader, opt, hyp_dict, device):
    for epoch in range(0, hyp_dict['epochs']):
        # Adjust learning rate
        adjust_learning_rate(optimizer=optimizer,
                             current_epoch=epoch + 1,
                             max_epoch=hyp_dict['epochs'],
                             lr0=hyp_dict['lr0'],
                             lrf=hyp_dict['lrf'],
                             warmup_epochs=hyp_dict['warmup_epochs'])
        fit_one_epoch(model, train_loader, val_loader, loss_func, epoch, hyp_dict['epochs'], opt, names, device)


if __name__ == '__main__':
    # Parameter settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-path', type=str, default='result', help='result storage address')
    parser.add_argument('--data', type=str, default='config/dataset.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='config/hyp.yaml', help='hyperparameters path')

    parser.add_argument('--pretrained', type=bool, default=False, help='whether start pre training?')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--single-cls', action='store_true', default=False,
                        help='train as single-class dataset')

    opt = parser.parse_args()

    # Dataset information loading
    with open(opt.data, encoding='utf-8') as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict
    nc, names = (1, ['item']) if opt.single_cls else (int(data_dict['nc']), data_dict['names'])  # number classes, names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # Loading hyperparameter information
    with open(opt.hyp, encoding='utf-8') as f:
        hyp_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get data
    train_loader, val_loader = get_train_dataloader(data_dict, names, hyp_dict)

    # Build model
    model = SCNet(nc=nc)
    model = config_model(model, opt, device)

    # loss function
    loss_func = nn.CrossEntropyLoss(weight=torch.tensor([4.6, 1.3, 1.1, 1.0, 1.1, 1.0, 1.0]).to(device))

    # optimizer
    optimizer = build_optimizer(model,
                                name=hyp_dict['optimizer'],
                                lr=hyp_dict['lr0'],
                                momentum=hyp_dict['momentum'])

    best_acc = 0.0

    # train
    train(model, loss_func, train_loader, val_loader, opt, hyp_dict, device)
