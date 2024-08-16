"(Convolutional) Neural Network Classification Model"

import os, argparse
import math
import numpy as np
import timeit

import torch
import tqdm
from torch.optim.lr_scheduler import MultiStepLR

from dataset import dataset
os.makedirs('./results', exist_ok=True)

def main(seed=2024):
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', nargs='?', type=str, default='cifar10')
    args = parser.parse_args()
    
    # Setting manual seed for reproducibility
    torch.manual_seed(seed)
    
    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using the '+device+' device...')
    
    # load data and network model
    train_loader, test_loader, model, num_classes = dataset(args.dataset_name, seed, batch_size=256)
    
    # set device
    model = model.to(device)
    
    # define the loss
    criterion = torch.nn.CrossEntropyLoss()
    
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    num_epochs = 1000#{'mnist':100, 'cifar10':500}[args.dataset_name]
    scheduler = MultiStepLR(optimizer, milestones=[0.5 * num_epochs, 0.75 * num_epochs], gamma=0.1)
    
    # define training and testing procedures
    def train(epoch):
        model.train()
    
        minibatch_iter = tqdm.tqdm(train_loader, desc=f"(Epoch {epoch}) Minibatch")
        for data, target in minibatch_iter:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            minibatch_iter.set_postfix(loss=loss.item())
        return loss.item()
    
    def test(epoch):
        model.eval()
    
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                pred = output.argmax(-1)
                correct += pred.eq(target.view_as(pred)).cpu().sum()
        acc = correct / float(len(test_loader.dataset))
        print('Epoch {}: test set: Accuracy: {}/{} ({}%)'.format(
            epoch, correct, len(test_loader.dataset), 100. * acc
        ))
        return acc.item()
    
    # Train the model
    os.makedirs('./results', exist_ok=True)
    loss_list = []
    acc_list = []
    times = np.zeros(2)
    for epoch in range(1, num_epochs + 1):
        beginning=timeit.default_timer()
        loss_list.append(train(epoch))
        times[0] += timeit.default_timer()-beginning
        beginning=timeit.default_timer()
        acc_list.append(test(epoch))
        times[1] += timeit.default_timer()-beginning
        # scheduler.step()
        state_dict = model.state_dict()
        torch.save({'model': state_dict}, os.path.join('./results','cnn_'+args.dataset_name+'_checkpoint.dat'))
    
    # save to file
    stats = np.array([acc_list[-1], times.sum()])
    stats = np.array(['CNN']+[np.array2string(r, precision=4) for r in stats])[None,:]
    header = ['Method', 'ACC', 'time']
    np.savetxt(os.path.join('./results',args.dataset_name+'_CNN.txt'),stats,fmt="%s",delimiter=',',header=','.join(header))
    np.savez_compressed(os.path.join('./results',args.dataset_name+'_CNN'), loss=np.stack(loss_list), acc=np.stack(acc_list), times=times)
    
    # plot the result
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2, figsize=(10,4))
    axes[0].plot(loss_list)
    axes[0].set_ylabel('Negative ELBO loss')
    axes[1].plot(acc_list)
    axes[1].set_ylabel('Accuracy')
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.savefig(os.path.join('./results',args.dataset_name+'_CNN.png'), bbox_inches='tight')

if __name__ == '__main__':
    main()