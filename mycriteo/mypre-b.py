#!/usr/bin/env python3

import argparse, csv, sys

from common import *

if len(sys.argv) == 1:
    sys.argv.append('-h')

from common import *

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nr_bins', type=int, default=int(1e+6))
parser.add_argument('-t', '--threshold', type=int, default=int(10))
parser.add_argument('-f', '--file_type', type=str, default='train')
parser.add_argument('-i', '--prod_num', type=str, default=0) # 0 ~ 23
parser.add_argument('csv_path', type=str)
parser.add_argument('gbdt_path', type=str)
parser.add_argument('out_path', type=str)
args = vars(parser.parse_args())

def gen_hashed_fm_feats(feats, nr_bins):
    feats = ['{0}:{1}:1'.format(field-1, hashstr(feat, nr_bins)) for (field, feat) in feats]
    return feats

frequent_feats = read_freqent_feats(args['threshold'], isTarget = True) # features that appears more than threshold times 

with open(args['out_path'], 'w') as f:
    count = 0
    count_pos = 0
    count_neg = 0
    count_total = 0

    for row, line_gbdt in zip(csv.DictReader(open(args['csv_path'])), open(args['gbdt_path'])):
        if count % 1000 == 0:
            print count
        count += 1

        # get label
        idx = int(args['prod_num'])
        label = int(row[TARGET[idx]])
        """
        if args['file_type'] == 'train':
            label = int(int(row['Label']) == idx)
        else:
            label = int(row[TARGET[idx]])
        """
        if label == 1:
            count_pos += 1
        else:
            count_neg += 1
        count_total += 1

        feats = []
        feat_idx = 1 
        for feat in gen_feats(row):
            field = feat.split('-')[0]
            if (field in FEAT_TARGET) or (field in FEAT_CAT) and (feat not in frequent_feats):
                feat = feat.split('-')[0]+'less'
            feats.append((feat_idx, feat))
            feat_idx += 1

        for i, feat in enumerate(line_gbdt.strip().split()[1:], start=1):
            feats.append((feat_idx, str(i)+":"+feat))

        feats = gen_hashed_fm_feats(feats, args['nr_bins']) # hash transformed feats
        f.write(str(label) + ' ' + ' '.join(feats) + '\n')

    print "count_pos = ", count_pos
    print "count_neg = ", count_neg
    print "count_total = ", count_total


