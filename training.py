import argparse
from modules.datasets import *
from modules.models import *
from modules.train import *
from sklearn.model_selection import KFold
from transformers import AutoTokenizer
from torch.optim.lr_scheduler import ExponentialLR

parser = argparse.ArgumentParser(description="Used to finetune a bert model for AFP.")
parser.add_argument('--model_name', type=str, required=True, help='The Bert Model Name')
parser.add_argument('--classifier_type', type=str, default='MLP')
parser.add_argument('--classifier_args', type=int, nargs='+', required=True)
parser.add_argument('--training_epochs', type=int, default=25)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--data_dir', type=str, default='data/Antifp_Main/')
parser.add_argument('--save_dir', type=str, default='models/')
parser.add_argument('--log_dir', type=str, default='models/')
parser.add_argument('--mask_ratio', type=float, default=0.)
parser.add_argument('--max_length', type=int, default=128)
parser.add_argument('--mask_token', type=int, default=32)
parser.add_argument('--use_feature', type=int, default=0)
args = parser.parse_args()

if __name__ == '__main__':
    device = 'cuda' if cuda.is_available() else 'cpu'
    model_name = args.model_name
    mask_token = args.mask_token
    if model_name == 'Rostlab/prot_bert':
        mask_token=1
    elif model_name == 'dmis-lab/biobert-base-cased-v1.2':
        mask_token=103
    print(mask_token)
    save_dir = os.path.join(args.save_dir, model_name)
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, args.model_name.split('/')[0])):
        os.mkdir(os.path.join(args.save_dir, args.model_name.split('/')[0]))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    log_dir = os.path.join(args.log_dir, model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
    seqs, labels, features = getAFP_Main_train(args.data_dir)
    inputs, lengths = encode(seqs, tokenizer, max_length=args.max_length)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for idx, split_index in enumerate(kf.split(lengths, labels)):
        if not os.path.exists(save_dir + f'/fold{idx}'):
            os.mkdir(save_dir + f'/fold{idx}')
        if not os.path.exists(log_dir + f'/fold{idx}'):
            os.mkdir(log_dir + f'/fold{idx}')
        model = sequenceClassifier(model_name, args.classifier_type, tuple(args.classifier_args)).to(device)
        model.embeds.model.encoder.requires_grad_(False)
        model.embeds.model.embeddings.requires_grad_(False)
        print('--------------------')
        print(f'       fold {idx + 1}')
        print('--------------------')
        train_idx, test_idx = split_index
        if args.use_feature:
            train_dataloader, test_dataloader = getDataloaderwithFeature(inputs, lengths, features, labels, train_idx, 
                                                                         test_idx, device, batch_size=args.batch_size, shuffle=True)
        else:
            train_dataloader, test_dataloader = getDataloader(
                inputs, lengths, labels, train_idx, test_idx, device, 
                batch_size=args.batch_size, shuffle=True 
         )
        loss_fn = torch.nn.functional.binary_cross_entropy
        optimizer = torch.optim.Adam([{'params': model.embeds.parameters(), 'lr': 1e-5},
                                    {'params': model.classifier.parameters(), 'lr':1e-4}],  weight_decay=1e-4)
        scheduler = ExponentialLR(optimizer, 0.999)
        fine_tune(model, args.training_epochs, train_dataloader, test_dataloader, loss_fn, optimizer, 
                  scheduler, save_dir=save_dir + f'/fold{idx}/', log_dir=log_dir + f'/fold{idx}/',
                  mask_ratio=args.mask_ratio, mask_token=mask_token, use_feature=args.use_feature)
    print('Done!')
        
