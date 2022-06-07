import numpy as np


def str2label(value, char2idx):
    label = [char2idx[v] for v in value if v in char2idx]
    return np.array(label, np.int32)


def label2str(label, idx2char, raw, blank_char="~"):
    blank = len(idx2char)
    if raw:
        return ''.join(blank_char if c == blank else idx2char[c] for c in label)
    return ''.join(idx2char[c] for c in label if c != blank)


def best_path_decode(output, blank_class):
    # type: () -> Tuple[..., List[...]]
    raw_pred = np.argmax(output, axis=1)

    mask = np.ones(len(raw_pred), dtype=bool)
    mask[1:] = (raw_pred[1:] != raw_pred[:-1])
    mask &= (raw_pred != blank_class)
    pred_data = raw_pred[mask]

    return pred_data, raw_pred
