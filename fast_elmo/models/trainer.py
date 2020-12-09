from tqdm import tqdm
import torch

tqdm_bar_format = "{l_bar}{bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] | {postfix[0]} {postfix[1][value]:>8.4g}                                 "


def train_language_model(model, loss_forward, loss_backward, optimizer, dataloader):
    model.train()

    losses = []
    with tqdm(total=len(dataloader), bar_format=tqdm_bar_format, leave=False,
              desc='Training language model', postfix=["Loss:", dict(value=0)]) as t:
        for batch in dataloader:
            optimizer.zero_grad()
            for key in batch.keys():
                batch[key] = batch[key].cuda()

            model_out = model(**batch)
            forward_out = model_out['forward_out'].flatten(0, 1)
            backward_out = model_out['backward_out'].flatten(0, 1)

            forward_target = batch['forward_target'].flatten()
            backward_target = batch['forward_target'].flip([1]).flatten()

            forward_loss_value = loss_forward(forward_out, forward_target)
            backward_loss_value = loss_backward(backward_out, backward_target)

            forward_loss_value.loss.backward()
            backward_loss_value.loss.backward()

            total_loss_value = (forward_loss_value.loss.item() + backward_loss_value.loss.item()) / 2

            losses.append(total_loss_value)
            t.postfix[1]["value"] = total_loss_value
            t.update()

            optimizer.step()
    return losses


def evaluate_language_model(model, loss, dataloader):
    model.eval()
    perplexies = []
    losses = []
    with torch.no_grad():
        with tqdm(total=len(dataloader), bar_format=tqdm_bar_format, leave=False,
                  desc='Evaluating language model', postfix=["Loss:", dict(value=0)]) as t:
            for batch in dataloader:
                for key in batch.keys():
                    batch[key] = batch[key].cuda()

                ids = batch['ids']
                mask = batch['mask']
                target = batch['forward_target'].flatten()

                model_out = model.forward_lm(ids, mask)
                model_out_flatten = model_out.flatten(0, 1)

                loss_value = loss(model_out_flatten, target)
                losses.append(loss_value.loss.item())
                t.postfix[1]["value"] = loss_value.loss.item()
                t.update()

                for idx in range(model_out.shape[0]):
                    m_out = model_out[idx]
                    msk = mask[idx]
                    tgs = batch['forward_target'][idx]

                    logprobs = loss.log_prob(m_out).cpu()
                    n = msk.sum().item()
                    perplexity = logprobs[torch.arange(tgs.shape[0]), tgs] * (-1 / n)
                    perplexity = perplexity[msk].sum().exp().item()
                    perplexies.append(perplexity)

    return losses, perplexies
