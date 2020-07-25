import torch


def train_language_model(model, loss, optimizer, dataloader):
    model.train()

    losses = []
    for batch in dataloader:
        optimizer.zero_grad()

        ids, mask, targets = batch

        ids = ids.cuda()
        mask = mask.cuda()
        target = targets.flatten().cuda()

        model_out = model(ids, mask)
        model_out_flatten = model_out.flatten(0, 1)
        loss_value = loss(model_out_flatten, target)
        loss_value.loss.backward()

        losses.append(loss_value.loss.item())

        optimizer.step()
    return losses


def evaluate_language_model(model, loss, dataloader):
    model.eval()

    perplexies = []
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            ids, mask, targets = batch

            ids = ids.cuda()
            mask = mask.cuda()
            target = targets.flatten().cuda()

            model_out = model(ids, mask)
            model_out_flatten = model_out.flatten(0, 1)
            loss_value = loss(model_out_flatten, target)
            losses.append(loss_value.loss.item())

            for idx in range(model_out.shape[0]):
                m_out = model_out[idx]
                msk = mask[idx]
                tgs = targets[idx]

                logprobs = loss.log_prob(m_out).cpu()
                n = msk.sum().item()
                perplexity = logprobs[torch.arange(tgs.shape[0]), tgs] * (-1 / n)
                perplexity = perplexity[msk].sum().exp().item()
                perplexies.append(perplexity)

    return losses, perplexies