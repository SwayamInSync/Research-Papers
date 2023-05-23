import os
import torch
import torch.nn as nn
import torch.optim as optim
from model import Word2Vec
from dataloader import get_word_loader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def train(model, criterion, optimizer, dataloader, num_epochs, device):
    step = 1
    writer = SummaryWriter('logs')
    for epoch in range(1, num_epochs+1):
        total_loss = 0
        loop = tqdm(dataloader, total=len(dataloader), leave=False)
        for center, context in loop:
            center, context = center.to(device), context.to(device)
            preds = model(center)
            loss = criterion(preds, context)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_description(f"{epoch}/{num_epochs}")
            loop.set_postfix({'step': step, 'loss': loss.item()})
            step += 1
            if step % 1000 == 0:
                loop.set_description("model saving")
                checkpoint = {
                    'state_dict': model.module.state_dict,
                    'optimizer_state': optimizer.state_dict,
                    'criterion_state': criterion.state_dict
                }
                torch.save(checkpoint, f"checkpoints/{step}_{loss}.pt")
            total_loss += loss.item()
            del loss, preds
        total_loss /= len(dataloader)
        writer.add_scalar('loss', total_loss, global_step=step)


dataset, dataloader = get_word_loader(batch_size=512, shuffle=True)
# hyper parameters
num_embeddings = len(dataset.vocab)
embedding_dim = 300
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Word2Vec(num_embeddings, embedding_dim).to(device)
model = torch.nn.DataParallel(model, [0, 1, 2])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=0.003)

train(model, criterion, optimizer, dataloader, 100, device)
