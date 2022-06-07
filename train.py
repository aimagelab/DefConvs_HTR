import sys
import os
from dataset.dataset import HwLineDataset
from models import crnn, vgg
import torch
from torch.utils.data import DataLoader
from torch.nn import CTCLoss
from utils import metrics, string_utils
from dataset import dataset
import argparse
import numpy as np
import json
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from torchvision import transforms
from utils.transforms import Resize
import msgpack



def set_random_seed(seed):
    # type: (int) -> None
    torch.manual_seed(seed)
    np.random.seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # random.seed(seed)


def create_dir(dir_name):
    # type: (str) -> None
    try:
        os.makedirs(dir_name)
    except FileExistsError:
        pass


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--img_root_dir', type=str, default='lines/')
    parser.add_argument('--train_set_fname', type=str, default='train.msgpack')
    parser.add_argument('--val_set_fname', type=str, default='val.msgpack')
    parser.add_argument('--test_set_fname', type=str, default='test.msgpack')
    parser.add_argument('--charset_fname', type=str, default='charset.msgpack')
    parser.add_argument('--data_augmentation', default=False, action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--img_height', type=int, default=60)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--cnn_output_size', type=int, default=1024)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--model_save_dir', type=str, default='../models/')
    parser.add_argument('--tensorboard_logs_dir', type=str, default='tensorboard_logs/')
    parser.add_argument('--patience', type=int, default=20)  # num epochs
    parser.add_argument('--cnn', type=str, default='cnn')
    parser.add_argument('--rnn', type=str, default='blstm')
    parser.add_argument('--deform_layers', type=str, default='0,1,2,3,4,5,6')  # NOT USED
    return parser.parse_args()


def main():
    # type: () -> int

    os.environ['OMP_NUM_THREADS'] = "1"
    os.environ['MKL_NUM_THREADS'] = "1"
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    torch.set_num_threads(1)

    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.enabled = False

    args = parse_arguments()
    cfg = vars(args)

    model_name = cfg['model_name']
    print(model_name)

    model_save_dir = cfg['model_save_dir']
    create_dir(model_save_dir)

    cfg_path = os.path.join(model_save_dir, f'{model_name}.json')

    with open(cfg_path, 'w') as f:
        json.dump(cfg, f, indent=4)

    print(args)
    print(cfg)

    seed = cfg['seed']
    set_random_seed(seed)

    tb_logs_dir = os.path.join(model_save_dir, cfg['tensorboard_logs_dir'])
    create_dir(tb_logs_dir)
    writer = SummaryWriter(log_dir=os.path.join(tb_logs_dir, model_name))

    with open(os.path.join(cfg['data_path'], cfg['charset_fname']), 'rb') as f:
        charset = msgpack.load(f, strict_map_key=False, use_list=False)
        char2idx = charset['char2idx']
        idx2char = charset['idx2char']

    transforms_list = [Resize(cfg['img_height'])]

    transforms_list.append(transforms.ToTensor())

    if cfg['data_augmentation']:
        print('Data augmentation is enabled')
        transforms_list.append(

            transforms.RandomOrder([
                transforms.ColorJitter(brightness=(0.5, 5), contrast=(0.1, 10), saturation=(0, 5), hue=(-0.1, 0.1)),
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2)),
                transforms.RandomApply([
                    transforms.RandomAffine(degrees=(-1, 1), shear=(-50, 30)),
                    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                    transforms.RandomRotation(degrees=(-1, 1))]),
            ])
        )

    transforms_list.append(transforms.Normalize((0.5,) * 3, (0.5,) * 3, inplace=True))
    print(transforms_list)

    train_transforms = transforms.Compose(transforms_list)

    train_dataset = HwLineDataset(os.path.join(cfg['data_path'], cfg['train_set_fname']),
                                  char2idx,
                                  root_path=os.path.join(cfg['data_path'], cfg['img_root_dir']),
                                  transform=train_transforms)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=cfg['batch_size'],
                                  shuffle=True,
                                  num_workers=cfg['num_workers'],
                                  collate_fn=dataset.HwLineCollate(),
                                  pin_memory=True)

    val_transforms = transforms.Compose(transforms_list)

    val_dataset = HwLineDataset(os.path.join(cfg['data_path'], cfg['val_set_fname']),
                                char2idx,
                                root_path=os.path.join(cfg['data_path'], cfg['img_root_dir']),
                                transform=val_transforms)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=cfg['batch_size'],
                                shuffle=False,
                                num_workers=cfg['num_workers'],
                                collate_fn=dataset.HwLineCollate(),
                                pin_memory=True)

    alphabet_size = len(char2idx)

    model_dict = {
        'cnnd5l': crnn.CNND5L,
        'deformcnnd5l': crnn.DeformCNND5L,
        'cnn5l': crnn.CNN5L,
        'deformcnn5l': crnn.DeformCNN5L,
        'cnn': crnn.CNN,
        'cnn_regdo': crnn.CNN_regularized_do,
        'deformcnn': crnn.DeformCNN,
        'deformcnn_reg': crnn.DeformCNN_regularized,
        'deformcnn_regdo': crnn.DeformCNN_regularized_do,
        'blstmd5l': crnn.BLSTMD5L,
        'blstm': crnn.BLSTM
    }

    if cfg['cnn'] == 'vgg':
        cnn = vgg.vgg(vgg.mvgg11, vgg.dc,
                      vgg.bn, vgg.ms, vgg.ks, vgg.ss, vgg.ps)
    else:
        cnn = model_dict[cfg['cnn']](3)

    rnn = model_dict[cfg['rnn']](cfg['cnn_output_size'], cfg['hidden_size'], alphabet_size + 1)
    net = crnn.CRNN(cnn, rnn)
    print(net)

    if torch.cuda.is_available():
        net.cuda()
        dtype = torch.cuda.FloatTensor
    else:
        sys.exit("CUDA not available")

    if (cfg['cnn'] == 'cnnd5l' or cfg['cnn'] == 'deformcnnd5l') and cfg['rnn'] == 'blstmd5l':
        optimizer = torch.optim.RMSprop(net.parameters(), lr=cfg['lr'])
        print('OPTIMIZER: Using RMSprop')
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=cfg['lr'])
        print('OPTIMIZER: Using Adam')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', threshold=0.05, verbose=True)

    criterion = CTCLoss(reduction='sum', blank=alphabet_size)

    lowest_cer = np.inf

    max_patience = cfg['patience']
    patience = 0

    start_epoch = 0

    last_model_path = os.path.join(model_save_dir, f"{model_name}_last.pt")
    if os.path.isfile(last_model_path):
        model = torch.load(last_model_path)
        net.load_state_dict(model['state_dict'], strict=False)
        optimizer.load_state_dict(model['optimizer'])
        start_epoch = model['epoch'] + 1
        patience = model['patience']
        lowest_cer = model['lowest_cer']

    num_imgs_train = len(train_dataset)
    num_imgs_val = len(val_dataset)
    num_batches = len(train_dataloader)


    for epoch in range(start_epoch, cfg['num_epochs'] + 1):

        print("Epoch".rjust(5),
              "Train CER".rjust(12),
              "Train WER".rjust(12),
              "Train CTC Loss".rjust(17)
              )

        train_sum_wer = 0
        train_sum_cer = 0
        train_sum_loss = 0

        net.train()

        batch_counter = 0
        for line_imgs, labels, label_lengths, gt_list, line_id in train_dataloader:
            batch_counter += 1
            line_imgs = line_imgs.type(dtype)

            optimizer.zero_grad()

            preds = net(line_imgs)

            with torch.no_grad():
                preds_lengths = (preds.size(0),) * preds.size(1)
                out = preds.permute(1, 0, 2).detach().cpu().numpy()

            loss = criterion(preds, labels, preds_lengths, label_lengths)

            with torch.no_grad():
                train_sum_loss += loss.item()

            loss.backward()
            optimizer.step()

            for logits, gt_line in zip(out, gt_list):
                pred, _ = string_utils.best_path_decode(logits, alphabet_size)
                pred_str = string_utils.label2str(pred, idx2char, False)
                cer = metrics.cer(gt_line, pred_str)
                train_sum_cer += cer
                wer = metrics.wer(gt_line, pred_str)
                train_sum_wer += wer

            train_cer_score = train_sum_cer / (batch_counter*cfg['batch_size'])
            train_wer_score = train_sum_wer / (batch_counter*cfg['batch_size'])
            train_loss = train_sum_loss / batch_counter

            if np.mod(batch_counter, cfg['batch_size']*10) == 0:
                print(str(epoch).rjust(5),
                      f'{round(train_cer_score, 3):.3f}'.rjust(12),
                      f'{round(train_wer_score, 3):.3f}'.rjust(12),
                      f'{round(train_loss, 6):.6f}'.rjust(17))

        val_sum_wer = 0
        val_sum_cer = 0
        val_sum_loss = 0

        net.eval()

        for line_imgs, labels, label_lengths, gt_list, line_id in val_dataloader:
            line_imgs = line_imgs.type(dtype)

            with torch.no_grad():
                preds = net(line_imgs)

                preds_lengths = (preds.size(0),) * preds.size(1)

                out = preds.permute(1, 0, 2).detach().cpu().numpy()

                loss = criterion(preds, labels, preds_lengths, label_lengths)
                val_sum_loss += loss.item()

            for logits, gt_line in zip(out, gt_list):
                pred, _ = string_utils.best_path_decode(logits, alphabet_size)
                pred_str = string_utils.label2str(pred, idx2char, False)
                cer = metrics.cer(gt_line, pred_str)
                val_sum_cer += cer
                wer = metrics.wer(gt_line, pred_str)
                val_sum_wer += wer

        train_cer_score = train_sum_cer / num_imgs_train
        train_wer_score = train_sum_wer / num_imgs_train
        train_loss = train_sum_loss / num_batches
        val_cer_score = val_sum_cer / num_imgs_val
        val_wer_score = val_sum_wer / num_imgs_val
        val_loss = val_sum_loss / num_batches

        print("Epoch: "+str(epoch).rjust(5),
              "Train CER: "+f'{round(train_cer_score, 3):.3f}'.rjust(12),
              "Valid CER: "+f'{round(val_cer_score, 3):.3f}'.rjust(12),
              "Train WER: "+f'{round(train_wer_score, 3):.3f}'.rjust(12),
              "Valid WER: "+f'{round(val_wer_score, 3):.3f}'.rjust(12),
              "Train CTC Loss: "+f'{round(train_loss, 6):.6f}'.rjust(17),
              "Valid CTC Loss: "+f'{round(val_loss, 6):.6f}'.rjust(17))

        writer.add_scalar('Training/CER', train_cer_score, epoch)
        writer.add_scalar('Training/WER', train_wer_score, epoch)
        writer.add_scalar('Training/CTC_Loss', train_loss, epoch)
        writer.add_scalar('Validation/CER', val_cer_score, epoch)
        writer.add_scalar('Validation/WER', val_wer_score, epoch)
        writer.add_scalar('Validation/CTC_Loss', val_loss, epoch)

        best = False
        patience += 1
        if lowest_cer > val_cer_score + 0.0005:
            patience = 0
            lowest_cer = val_cer_score
            best = True

        torch.save({
            'epoch': epoch,
            'patience': patience,
            'lowest_cer': lowest_cer,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, last_model_path)

        if best:
            copyfile(last_model_path, os.path.join(model_save_dir, f"{model_name}_best.pt"))

        if patience >= max_patience:
            print("Reached", max_patience, "epochs without improvement.")
            writer.add_text('End', f"Reached {max_patience} epochs without improvement.")
            writer.add_text('Best Validation CER', f'Best CER score: {lowest_cer}')
            writer.close()
            break

        scheduler.step(val_cer_score)

    writer.add_text('Best Validation CER', f'Best CER score: {lowest_cer}')
    writer.close()


if __name__ == "__main__":
    main()
