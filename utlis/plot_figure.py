import matplotlib.pyplot as plt
import pandas as pd


def plot_figure(dir_path):
    dataframe = pd.read_csv(f'{dir_path}/loss_acc.csv')
    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    plt.rcParams['lines.linewidth'] = 0.9
    header = dataframe.columns
    for i in header:
        y = dataframe[i].tolist()
        x = range(1, len(y) + 1)

        flg = plt.figure(figsize=(6, 5), dpi=300)
        plt.plot(x, y, linestyle='-', color="red")
        plt.xlabel('epoch', fontdict={'size': 16})
        plt.ylabel(i, fontdict={'size': 16})
        plt.savefig(f'{dir_path}/{i}.jpg')
        plt.close()