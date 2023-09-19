import argparse

import matplotlib.pyplot as plt
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from utlis.datasets import *
from utlis.datasets import MyDataset
from utlis.model import SCNet

if __name__ == '__main__':
    # Parameter settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='config/dataset.yaml', help='dataset config file path')
    parser.add_argument('--hyp', type=str, default='config/hyp.yaml', help='hyperparameters path')

    parser.add_argument('--output-path', type=str, default='result/', help='output result path')
    parser.add_argument('--weights-path', type=str, default='result/SCNet_gri.pth')
    parser.add_argument('--img-size', nargs='+', type=int, default=[64, 64], help='[wide, height]')
    parser.add_argument('--batch-size', type=int, default=128, help='total batch size for all GPUs')
    opt = parser.parse_args()

    # Dataset information loading
    with open(opt.data, encoding='utf-8') as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict

    # Loading hyperparameter information
    with open(opt.hyp, encoding='utf-8') as f:
        hyp_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict
        hyp_dict["augment"] = False  # close augment

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get data
    test_path = data_dict['test']
    test_files = get_file_paths(test_path)
    test_dataset = MyDataset((test_files, 'folder'), names=data_dict['names'], hyp_dict=hyp_dict)
    test_loader = dataloader = DataLoader(dataset=test_dataset, num_workers=2, batch_size=opt.batch_size)
    # loading model
    net = SCNet(nc=data_dict['nc']).eval()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = torch.nn.DataParallel(net)
    state_dict = torch.load(opt.weights_path, map_location=device)
    net.load_state_dict(state_dict, strict=False)
    net = net.cuda()

    # prediction
    y_pred, y_ture = [], []
    for iteration, batch in tqdm(enumerate(test_loader)):
        datas, labels = batch[0], batch[1]
        labels = labels.cuda()

        outputs = net(datas)
        y_pred += outputs.argmax(dim=1).cpu()
        y_ture += labels.cpu()

    # Calculate various indicators and save them
    os.makedirs(os.path.join(opt.output_path, 'test'), exist_ok=True)
    result = classification_report(y_ture, y_pred, digits=4, target_names=data_dict['names'])
    with open(f'{opt.output_path}/test/metric.txt', 'w') as f:
        f.write(result)
    print(result)

    # Calculate the confusion matrix and save the drawing
    cm = confusion_matrix(y_ture, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data_dict['names'])
    fig, ax = plt.subplots(dpi=300)
    disp.plot(cmap='Blues', ax=ax)
    plt.savefig(f'{opt.output_path}/test/confusion_matrix.jpg')
    print(cm)
