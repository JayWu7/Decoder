import re


def read_output(filename, out_path):
    wrong_dict = {}
    total_phone = 0
    total_wrong = 0
    correct_phones = {'x', 'iao3', 'q', 'ing1', 'sil'}
    re_pattern = re.compile("\'([a-z]*\d?)\'")
    with open(filename, 'r') as f:
        for row in f.readlines():
            if row.startswith('deque'):
                phones = re_pattern.findall(row)
                for pho in phones:
                    total_phone += 1
                    if pho not in correct_phones:
                        total_wrong += 1
                        if pho in wrong_dict:
                            wrong_dict[pho] += 1
                        else:
                            wrong_dict[pho] = 1

    with open(out_path, 'w') as f:
        f.write('Total Phone amount is {}\n'.format(total_phone))
        f.write('Total wrong amount is {}, wrong rate is {:.2}\n'.format(total_wrong, total_wrong / total_phone))
        f.write('Total wrong phone amount is {}\n\n'.format(len(wrong_dict)))
        f.write('Wrong phone distribution:\n')
        wrong_list = list(wrong_dict.items())
        wrong_list.sort(key=lambda x: x[1], reverse=True)

        for w_pho, amount in wrong_list:
            f.write('{}, {}\n'.format(w_pho, amount))


def detect_wrong(phones_file, ids_file, out_file):
    seqs_pho = []
    seqs_ids = []
    re_pattern_pho = re.compile("\'([a-z]*\d?)\'")
    re_pattern_id = re.compile("([\d]+)")
    correct_phones = {'x', 'iao3', 'q', 'ing1', 'sil'}

    with open(phones_file, 'r') as fp:
        for row in fp.readlines():
            if row.startswith('deque'):
                phones = re_pattern_pho.findall(row)
                seqs_pho.append(phones)

    with open(ids_file, 'r') as fi:
        for row in fi.readlines():
            if row.startswith('deque'):
                ids = re_pattern_id.findall(row)
                seqs_ids.append(ids)

    with open(out_file, 'w') as f:
        f.write('Wrong paths are: \n\n')
        wrong_len = 0
        for i, seq in enumerate(seqs_pho):
            for pho in seq:
                if pho not in correct_phones:
                    wrong_len += 1
                    f.write('Audio id: {}\n'.format(i))
                    f.write('{}\n'.format(seq))
                    f.write('{}\n\n'.format(seqs_ids[i]))
                    break
        f.write('Total wrong path amount is: {}'.format(wrong_len))


if __name__ == '__main__':
    # read_output('./out_full.txt', './detect_out_full.txt')
    detect_wrong('./out_full.txt', './out_full_ids.txt', 'wrong_path.txt')
