import threading
import time

from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator


def plot_training_history(iters, cur_loss, epochs, mean_iou):
    # fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    # ax.plot(history['epochs'], history['Overall_ACC'], label='Overall_ACC')
    # ax.plot(history['epochs'], history['Mean ACC'], label='Mean ACC')
    # ax.plot(history['epochs'], history['FreqW Acc'], label='FreqW Acc')
    # ax.plot(history['epochs'], history['Mean IoU'], label='Mean IoU')
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))

    ax1.plot(iters, cur_loss)
    ax1.set_xlabel('iter')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curve')
    ax1.set_ylim([0, 0.05])

    ax2.plot(epochs, mean_iou)
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('IoU')
    ax2.set_title('IoU Curve')
    ax2.set_ylim([0.8, 1])

    # x_major_locator = MultipleLocator(1)
    # y_major_locator = MultipleLocator(0.05)
    # plt.gca().yaxis.set_major_locator(y_major_locator)
    # plt.gca().xaxis.set_major_locator(x_major_locator)

    # ax.set_ylim([0.8, 1])
    # ax.legend()
    # ax.set_ylabel('ACC')
    # ax.set_xlabel('Epoch')
    # fig.suptitle('Training History')

    plt.show()



    # if epochs >= 100:
    #     plt.show()
    # else:
    #     plt.show()
    #     time.sleep(3)
    #     plt.close()


