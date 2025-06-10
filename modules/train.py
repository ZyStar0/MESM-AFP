import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef, average_precision_score, precision_recall_curve
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class Accumulator():
    def __init__(self):
        self.preds = []
        self.labels = []

    def add(self, pred, label):
        self.preds.append(pred)
        self.labels.append(label)


    def cat(self):
        self.preds = torch.cat(self.preds, dim=0)
        self.labels = torch.cat(self.labels, dim=0)

    def clear(self):
        self.preds = self.preds.detach().cpu().numpy().reshape(-1)
        self.labels = self.labels.detach().cpu().numpy().reshape(-1)

    def evaluate(self, info_print=False):
        """
        :return: accuracy f1 recall precision
        """
        if type(self.preds) == list:
            self.cat()
            self.clear()
        auc = roc_auc_score(self.labels, self.preds)
        aupr = average_precision_score(self.labels, self.preds)
        self.labels = self.labels.astype(np.int32)
        self.preds = np.round(self.preds).astype(np.int32)
        mcc = matthews_corrcoef(self.labels, self.preds)
        self.preds = np.round(self.preds)
        tn, fp, fn, tp = confusion_matrix(self.labels, self.preds, labels=[0, 1]).ravel()
        acc = (tp + tn) / (tp + fp + fn + tn)
        f1 = 2 * tp / (2 * tp + fp + fn)
        rec = tp / (tp + fn)
        prec = tp / (tp + fp)
        if info_print:
            print(f'tn:{tn} fp:{fp} fn:{fn} tp:{tp} acc:{acc:.4f} f1:{f1:.4f} rec:{rec:.4f} prec:{prec:.4f} auc:{auc:.4f} aupr:{aupr:.4f} mcc:{mcc:.4f}')
        return acc, f1, rec, prec, auc, aupr, mcc

    def print(self):
        acc, f1, rec, prec, auc, aupr, mcc = self.evaluate()
        print('Accuracy: %.4f' % acc)
        print('F1: %.4f' % f1)
        print('Recall: %.4f' % rec)
        print('Precision: %.4f' % prec)
        print('AUC: %.4f' % auc)
        print('AUPR: %.4f' % aupr)
        print('MCC: %.4f' % mcc)

# print the metrics to tensorboard
def log(writer, title, epoch, eval, loss=None):
    writer.add_scalar(title + '_acc', eval[0], epoch)
    writer.add_scalar(title + '_f1', eval[1], epoch)
    writer.add_scalar(title + '_rec', eval[2], epoch)
    writer.add_scalar(title + '_prec', eval[3], epoch)
    writer.add_scalar(title + '_auc', eval[4], epoch)
    writer.add_scalar(title + '_aupr', eval[5], epoch)
    writer.add_scalar(title + '_mcc', eval[6], epoch)
    if loss is not None:
        writer.add_scalar(title + '_loss', loss, epoch)


def fine_tune(model, epochs, train_dataloader, test_dataloader, loss_fn, optimizer, 
              scheduler=None, save_dir='', log_dir='', mask_ratio=None, mask_token=32, use_features=False):
    max_aupr = 0.60
    writer = SummaryWriter(log_dir=log_dir)
    patience = 5
    for epoch in tqdm(range(epochs)):
        if epoch == epochs // 5:
            model.embeds.model.encoder.layer[-5:].requires_grad_(True)
        # choose phase to train the model
        print('Epoch %d' % (epoch + 1))
        eval, loss = train(model, train_dataloader, loss_fn, optimizer,
                           scheduler, mask_ratio, mask_token, use_features)
        log(writer, 'train', epoch, eval, loss)
        print()
        eval, _ = predict(model, test_dataloader, use_features)
        log(writer, 'test', epoch, eval, loss)
        if eval[5] > max_aupr:
            patience == 5
            max_aupr = eval[5]
            torch.save(model.state_dict(), save_dir + f'/checkpoints_best.pth')
        else:
            patience -= 1
            if patience == 0:
                break
        print()

def mask(input_ids, length, mask_ratio, mask_token):
    mask_length = int(mask_ratio * length)
    mask_indices = torch.randperm(int(length))[:mask_length]
    input_ids[mask_indices] = mask_token
    return input_ids

def train(model, dataloader, loss_fn, optimizer, scheduler, mask_ratio=None, mask_token=32, use_feature=False):
    model.train()
    accum = Accumulator()
    total_loss = 0.0
    for batch in dataloader:
        if mask_ratio is not None:
            for i in range(len(batch[0])):
                batch[0][i] = mask(batch[0][i], batch[2][i], mask_ratio, mask_token=mask_token)
        optimizer.zero_grad()
        if use_feature:
            output = model(batch[:-2], batch[-2])
        else:
            output = model(batch[:-1])
        labels = batch[-1]
        loss = loss_fn(output.view_as(labels), labels.to(torch.float32))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
        accum.add(output, labels)
    eval = accum.evaluate(True)
    return eval, total_loss / len(dataloader)

def predict(model, dataloader, use_feature):
    model.eval()
    accum = Accumulator()
    res = []
    with torch.no_grad():
        for batch in dataloader:
            torch.cuda.empty_cache()
            if use_feature:
                output = model(batch[:-2], batch[-2])
            else:
                output = model(batch[:-1])
            labels = batch[-1]
            accum.add(output, labels)
            res.append(output)
    eval = accum.evaluate(True)
    # accum.print()
    return eval, torch.cat(res)

def get_embeds(model, dataloader, use_feature):
    model.eval()
    embeds_list = []
    with torch.no_grad():
        for batch in dataloader:
            if use_feature:
                output = model.getEmbeds(batch[:-2]).to('cpu')
            else:
                output = model.getEmbeds(batch[:-1]).to('cpu')
            embeds_list.append(output)
    return torch.cat(embeds_list)
