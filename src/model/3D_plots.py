import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def extract_train_acc(filename):
    train_acc_values = []
    train_loss_values = []
    val_acc_values = []
    val_loss_values = []

    with open(filename, 'r') as file:
        for line in file:
            match_1 = re.search(r'train_acc: ([\d.]+)', line)
            match_2 = re.search(r'train_loss: ([\d.]+)', line)
            match_3 = re.search(r'val_acc: ([\d.]+)', line)
            match_4 = re.search(r'val_loss: ([\d.]+)', line)
            if match_1:
                train_acc_values.append(float(match_1.group(1)))
            if match_2:
                train_loss_values.append(float(match_2.group(1)))
            if match_3:
                val_acc_values.append(float(match_3.group(1)))
            if match_4:
                val_loss_values.append(float(match_4.group(1)))

    return train_acc_values, train_loss_values, val_acc_values, val_loss_values

if __name__ == "__main__":
    filename = "training_logs_mse.txt"
    train_acc_values, train_loss_values, val_acc_values, val_loss_values = extract_train_acc(filename)

    train_acc_values = np.array(train_acc_values)
    train_loss_values = np.array(train_loss_values)
    val_acc_values = np.array(val_acc_values)
    val_loss_values = np.array(val_loss_values)

    # reshape arrays to 10x50 for 3D plots
    train_acc = train_acc_values.reshape(10, 50)
    train_loss = train_loss_values.reshape(10, 50)
    val_acc = val_acc_values.reshape(10, 50)
    val_loss = val_loss_values.reshape(10, 50)

    # grid
    x = np.arange(train_acc.shape[1])  # steps
    y = np.arange(train_acc.shape[0])  # epochs
    X, Y = np.meshgrid(x, y)

    epoch_ticks = y
    epoch_labels = epoch_ticks * 10  # display y_labels like [0, 10, 20, ..., 100]
    display_ticks = epoch_ticks[::2]  # display 20 by 20 (0, 20, 40, ..., 100)
    display_labels = epoch_labels[::2]

    fig = plt.figure(figsize=(20, 15))

    # train_acc
    ax1 = fig.add_subplot(221, projection='3d')
    surf1 = ax1.plot_surface(X, Y, train_acc, cmap='viridis', edgecolor='none', alpha=0.9)
    ax1.contour(X, Y, train_acc, zdir='z', offset=train_acc.min(), cmap='viridis', linestyles="solid")
    ax1.set_title('Training accuracy', fontsize=20)
    ax1.set_xlabel('Linear probing epochs', fontsize=14)
    ax1.set_ylabel('I-JEPA epochs', fontsize=14)
    ax1.set_zlabel('Accuracy', fontsize=14)
    ax1.set_yticks(display_ticks)
    ax1.set_yticklabels(display_labels)
    ax1.grid(True)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5, pad=0.1)

    # train_loss
    ax2 = fig.add_subplot(222, projection='3d')
    surf2 = ax2.plot_surface(X, Y, train_loss, cmap='plasma', edgecolor='none', alpha=0.9)
    ax2.contour(X, Y, train_loss, zdir='z', offset=train_loss.min(), cmap='plasma', linestyles="solid")
    ax2.set_title('Training loss', fontsize=20)
    ax2.set_xlabel('Linear probing epochs', fontsize=14)
    ax2.set_ylabel('I-JEPA epochs', fontsize=14)
    ax2.set_zlabel('Loss', rotation=90, fontsize=14)
    ax2.set_yticks(display_ticks)
    ax2.set_yticklabels(display_labels)
    ax2.grid(True)
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5, pad=0.1)

    # val_acc
    ax3 = fig.add_subplot(223, projection='3d')
    surf3 = ax3.plot_surface(X, Y, val_acc, cmap='magma', edgecolor='none', alpha=0.9)
    ax3.contour(X, Y, val_acc, zdir='z', offset=val_acc.min(), cmap='magma', linestyles="solid")
    ax3.set_title('Validation accuracy', fontsize=20)
    ax3.set_xlabel('Linear probing epochs', fontsize=14)
    ax3.set_ylabel('I-JEPA epochs', fontsize=14)
    ax3.set_zlabel('Accuracy', fontsize=14)
    ax3.set_yticks(display_ticks)
    ax3.set_yticklabels(display_labels)
    ax3.grid(True)
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5, pad=0.1)

    # val_loss
    ax4 = fig.add_subplot(224, projection='3d')
    surf4 = ax4.plot_surface(X, Y, val_loss, cmap='inferno', edgecolor='none', alpha=0.9)
    ax4.contour(X, Y, val_loss, zdir='z', offset=val_loss.min(), cmap='inferno', linestyles="solid")
    ax4.set_title('Validation loss', fontsize=20)
    ax4.set_xlabel('Linear probing epochs', fontsize=14)
    ax4.set_ylabel('I-JEPA epochs', fontsize=14)
    ax4.set_zlabel('Loss', rotation=90, fontsize=14)
    ax4.set_yticks(display_ticks)
    ax4.set_yticklabels(display_labels)
    ax4.grid(True)
    fig.colorbar(surf4, ax=ax4, shrink=0.5, aspect=5, pad=0.1)

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.savefig("linear_prob_evolution.png")
