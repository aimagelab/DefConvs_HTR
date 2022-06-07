import msgpack

def create_charset(subsets_path):
    subsets = ['train', 'test']
    char = set()
    for sset in subsets:
        subset_msgpack = subsets_path + sset + '.msgpack'
        subset_list = msgpack.load(open(subset_msgpack, 'rb'), use_list=False)
        for item in subset_list:
            gt = item[1]
            for c in gt:
                char.add(c)
    return sorted(list(char))


if __name__ == "__main__":
    chars_list = create_charset('/folder/name/')
    print(len(chars_list), 'characters: ', chars_list)
    charset = {}
    charset['char2idx'] = dict([(c, i) for c, i in zip(chars_list, range(len(chars_list)))])
    charset['idx2char'] = dict([(i, c) for i, c in zip(range(len(chars_list)), chars_list)])
    msgpack.dump(charset, open('/folder/name/charset.msgpack', 'wb'))