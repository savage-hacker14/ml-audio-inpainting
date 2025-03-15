# Imports
from tqdm import tqdm
import configparser
import torch


if __name__ == "__main__":
    # Load config file
    config = configparser.ConfigParser()
    config.read('blstm.config')

    
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        running_loss_ffn = 0.0

        for data in tqdm(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # Flatten inputs for ffn
            labels_one_hot = torch.tensor(F.one_hot(labels, num_classes=10), dtype=torch.float32)
            inputs_flattened =  torch.flatten(inputs, start_dim=1)

            # zero the parameter gradients
            optimizer_ffn.zero_grad()

            # forward + backward + optimize
            outputs = feedforward_net(inputs_flattened)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_ffn.step()
            running_loss_ffn += loss.item()

        # Store training loss
        ffn_loss[epoch] = running_loss_ffn

        print(f"Training loss: {running_loss_ffn}")

    print('Finished Training')

    torch.save(feedforward_net.state_dict(), 'ffn.pth')  # Saves model file (upload with submission)