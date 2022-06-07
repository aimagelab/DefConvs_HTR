import msgpack
import os
import sys
from dataset.dataset import HwLineDataset
from models import crnn, vgg
import torch
from torch.utils.data import DataLoader
from utils import metrics, string_utils
from dataset import dataset
import argparse
import json
from utils.transforms import Resize
from torchvision import transforms


def parse_arguments():
    # type: () -> argparse.Namespace
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--img_root_dir', type=str, default='lines/')
    parser.add_argument('--test_set_fname', type=str, default='test.msgpack')
    parser.add_argument('--charset_fname', type=str, default='charset.msgpack')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--img_height', type=int, default=60)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--cnn_output_size', type=int, default=1024)
    parser.add_argument('--model_save_dir', type=str, default='../models/')
    parser.add_argument('--cnn', type=str, default='cnn')
    parser.add_argument('--rnn', type=str, default='blstm')
    parser.add_argument('--deform_layers', type=str, default='0,1,2,3,4,5,6')  # NOT USED
    return parser.parse_args()


def main():
    # type: () -> None
    # os.environ["OMP_NUM_THREADS"] = "1"
    # os.environ["KML_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    args = parse_arguments()
    cfg_this = vars(args)

    model_name = cfg_this['model_name']
    print(model_name)

    model_save_dir = cfg_this['model_save_dir']

    if args.model_path:
        cfg_path = os.path.join(args.model_path, f'{model_name}.json')
    else:
        cfg_path = os.path.join(model_save_dir, f'{model_name}.json')

    if os.path.isfile(cfg_path):
        with open(cfg_path, 'r') as f:
            cfg = json.load(f)
    else:
        sys.exit('missing config file')

    with open(os.path.join(cfg['data_path'], cfg['charset_fname']), 'rb') as f:
        charset = msgpack.load(f, strict_map_key=False, use_list=False)
        char2idx = charset['char2idx']
        idx2char = charset['idx2char']

    test_transforms = transforms.Compose([Resize(cfg['img_height']),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5,) * 3, (0.5,) * 3, inplace=True)])


    test_dataset = HwLineDataset(os.path.join(cfg['data_path'], cfg_this['test_set_fname']),
                                 char2idx,
                                 root_path=os.path.join(cfg['data_path'], cfg['img_root_dir']),
                                 transform=test_transforms)
    test_dataloader = DataLoader(test_dataset,
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

    default_model_path = os.path.join(cfg['model_save_dir'], f"{model_name}_best.pt")
    arg_model_path = args.model_path
    if arg_model_path:
        model_path = os.path.join(arg_model_path, f"{model_name}_best.pt")
        model = torch.load(model_path)
        net.load_state_dict(model['state_dict'], strict=False)
    elif os.path.isfile(default_model_path):
        model = torch.load(default_model_path)
        net.load_state_dict(model['state_dict'], strict=False)

    sum_cer = 0
    sum_wer = 0
    sum_nncer = 0
    sum_nnwer = 0
    sum_clens = 0
    sum_wlens = 0

    num_imgs_test = len(test_dataset)

    net.eval()

    for line_imgs, _, _, gt_list, line_ids in test_dataloader:
        line_imgs = line_imgs.type(dtype)

        with torch.no_grad():
            preds = net(line_imgs).cpu()

            out = preds.permute(1, 0, 2).detach().cpu().numpy()

        for logits, gt_line, line_id in zip(out, gt_list, line_ids):
            pred, _ = string_utils.best_path_decode(logits, alphabet_size)
            pred_str = string_utils.label2str(pred, idx2char, False)
            print(line_id, '--->', pred_str)
            cer = metrics.cer(gt_line, pred_str)
            sum_cer += cer
            sum_nncer += metrics.nn_cer(gt_line, pred_str)
            sum_clens += len(gt_line)
            wer = metrics.wer(gt_line, pred_str)
            sum_wer += wer
            sum_nnwer += metrics.nn_wer(gt_line, pred_str)
            sum_wlens += len(gt_line.split(' '))

    print("Test CER", round((sum_cer / num_imgs_test) * 100, 1))
    print("Test WER", round((sum_wer / num_imgs_test) * 100, 1))
    print("Test normalized_CER", round((sum_nncer / sum_clens) * 100, 1))
    print("Test normalized_WER", round((sum_nnwer / sum_wlens) * 100, 1))
    print("Parameters", sum(p.numel() for p in net.parameters()))


if __name__ == "__main__":
    main()