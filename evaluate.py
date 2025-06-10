from sklearn.manifold import TSNE
import argparse
import torch
from transformers import AutoTokenizer
import os
from modules.models import sequenceClassifier, sequenceClassifierwithFeature
from modules.train import predict, get_embeds
from modules.datasets import getAFP_Main, getDataloader, encode
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Used to predict AFP.")
parser.add_argument('--model_name', type=str, required=True, 
                    help='The Bert Model Name')
parser.add_argument('--classifier_type', type=str, default='MLP')
parser.add_argument('--classifier_args', type=int, nargs='+', required=True)
parser.add_argument('--data_dir', type=str, default='data/Antifp_Main')
parser.add_argument('--output_dir', type=str, default='outputs')
parser.add_argument('--model_dir', type=str, default='models')
parser.add_argument('--fold', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_length', type=int, default=128)
parser.add_argument('--use_feature', type=int, default=0)
parser.add_argument('--feature_nums', type=int, default=0)
args = parser.parse_args()

if __name__ == '__main__':
    torch.no_grad()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = os.path.join(args.output_dir, args.model_name)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    if not os.path.exists(os.path.join(args.output_dir, args.model_name.split('/')[0])):
        os.mkdir(os.path.join(args.output_dir, args.model_name.split('/')[0]))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if args.use_feature:
        model = sequenceClassifierwithFeature(args.model_name, args.classifier_type, args.classifier_args, args.feature_nums).to(device).eval()
    else:
        model = sequenceClassifier(args.model_name, args.classifier_type, args.classifier_args).to(device).eval()
    best_model, best_aupr = '', 0.
    train_seqs, train_labels, test_seqs, test_labels = getAFP_Main(args.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataloader = getDataloader(*encode(train_seqs, tokenizer, args.max_length), train_labels, batch_size=args.batch_size, device=device)
    test_dataloader = getDataloader(*encode(test_seqs, tokenizer, args.max_length), test_labels, batch_size=args.batch_size, device=device)
    f = open(os.path.join(output_dir, 'eval.txt'), 'w')
    print('Reduce dim for base model')
    plt.figure(figsize=(13, 12))
    plt.suptitle('BCE Visualization', fontsize=28)
    train_embeds = get_embeds(model, train_dataloader).detach().to('cpu')
    torch.cuda.empty_cache()
    test_embeds = get_embeds(model, test_dataloader).detach().to('cpu')
    torch.cuda.empty_cache()
    tsne_train = TSNE(n_components=2, learning_rate=200).fit_transform(train_embeds.numpy(), train_labels)
    tsne_test = TSNE(n_components=2, learning_rate=200).fit_transform(test_embeds.numpy(), test_labels)
    plt.subplot(221)
    plt.scatter(tsne_train[:, 0], tsne_train[:, 1], c=train_labels)
    plt.title('Train Base', fontsize=20)
    plt.subplot(222)
    plt.scatter(tsne_test[:, 0], tsne_test[:, 1], c=test_labels)
    plt.title('Eval Base', fontsize=20)
    for idx in range(args.fold):
        f.write(f'fold{idx}:\n')
        print(f'fold{idx}:')
        model_dir = os.path.join(args.model_dir, args.model_name, f'fold{idx}', 'checkpoints_best.pth')
        model.to('cpu')
        state_dict = torch.load(model_dir, map_location='cpu')
        model.load_state_dict(state_dict)
        model.to(device).eval()
        eval, _ = predict(model, test_dataloader)
        if eval[5] > best_aupr:
            best_aupr = eval[5]
            best_model = model_dir
        for j, metric in enumerate(['acc', 'f1', 'rec', 'prec', 'auc', 'aupr', 'mcc']):
            f.write(f'{"{:<5}".format(metric)}:{eval[j]:.4f}\n')
    f.write(best_model)
    f.write('\n')
    f.close()
    print(f'{best_model} {best_aupr}')
    model.to('cpu')
    state_dict = torch.load(best_model, map_location='cpu')
    model.load_state_dict(state_dict)
    model.to(device).eval()
    print('Reduce dim for finetuned model')
    train_embeds = get_embeds(model, train_dataloader).detach().to('cpu')
    test_embeds = get_embeds(model, test_dataloader).detach().to('cpu')
    tsne_train = TSNE(n_components=2, learning_rate=200).fit_transform(train_embeds, train_labels)
    tsne_test = TSNE(n_components=2, learning_rate=200).fit_transform(test_embeds, test_labels)
    plt.subplot(223)
    plt.scatter(tsne_train[:, 0], tsne_train[:, 1], c=train_labels)
    plt.title('Train Finetune', fontsize=20)
    plt.subplot(224)
    plt.scatter(tsne_test[:, 0], tsne_test[:, 1], c=test_labels)
    plt.title('Eval Finetune', fontsize=20)
    pic = args.model_name.split('/')[1]
    if 'no_mask' in output_dir:
        pic = f'{pic}_no_mask.png'
    else:
        pic = f'{pic}_mask.png'
    print(pic)
    plt.savefig(os.path.join(output_dir, pic))
    