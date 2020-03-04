import numpy as np
import torch
from tqdm import tqdm


def train_batch(model, train_loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0

    dataset_length = len(train_loader.dataset)
    progress_bar = tqdm(total=len(train_loader.dataset), desc='Epoch')

    train_losses = []

    for (tags, lem), y in train_loader:
        batch_size = y.shape[0]
        sequence_length = y.shape[1]

        tags = tags.view(sequence_length, batch_size).to(device)
        lem = lem.view(sequence_length, batch_size).to(device)
        y = y.view(sequence_length, batch_size).to(device)

        optimizer.zero_grad()

        decoder_vocab_size = model.decoder.output_size
        y_ohe = torch.nn.functional.one_hot(y, decoder_vocab_size).to(torch.float).to(device)

        output = model(tags, lem, y_ohe, batch_size, sequence_length, sequence_length, device)

        batch_loss = criterion(output.view(batch_size, decoder_vocab_size, sequence_length),
                               y.view(batch_size, sequence_length))

        loss = batch_loss.mean()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        progress_bar.set_postfix(train_loss=np.mean(train_losses))
        progress_bar.update(batch_size)

    progress_bar.close()
    return np.mean(train_losses)


def test(model, train_loader, criterion, device):
    model.eval()

    progress_bar = tqdm(total=len(train_loader.dataset), desc='Epoch')

    train_losses = []

    with torch.no_grad():
        for (tags, lem), y in train_loader:
            batch_size = y.shape[0]
            sequence_length = y.shape[1]

            tags = tags.view(sequence_length, batch_size).to(device)
            lem = lem.view(sequence_length, batch_size).to(device)
            y = y.view(sequence_length, batch_size).to(device)

            decoder_vocab_size = model.decoder.output_size
            y_ohe = torch.nn.functional.one_hot(y, decoder_vocab_size).to(torch.float).to(device)

            output = model(tags, lem, y_ohe, batch_size, sequence_length, sequence_length, device, teacher_forcing_ratio=0)

            batch_loss = criterion(output.view(batch_size, decoder_vocab_size, sequence_length),
                                   y.view(batch_size, sequence_length))

            loss = batch_loss.mean()
            train_losses.append(loss.item())

            progress_bar.set_postfix(train_loss=np.mean(train_losses))
            progress_bar.update(batch_size)

    progress_bar.close()
    return np.mean(train_losses)


def train(model, n_epochs, train_loader, test_loader, validation_loader, device):
    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    optimizer = torch.optim.Adam(params=model.parameters())

    for epoch in range(n_epochs):
        train_loss = train_batch(model, train_loader, optimizer, criterion, device)
        valid_loss = test(model, validation_loader, criterion, device)
        print(f"epoch {epoch}: train_loss {train_loss}, valid_loss {valid_loss}")

    test_loss = test(model, test_loader, criterion, device)
    print(f"test loss {test_loss}")
