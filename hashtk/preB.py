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
parser.add_argument('csv_path', type=str)
parser.add_argument('gbdt_path', type=str)
parser.add_argument('out_path', type=str)
args = vars(parser.parse_args())

file_type = args['file_type']
def gen_hashed_fm_feats(feats, nr_bins):
    feats = ['{0}:{1}:1'.format(field-1, hashstr(feat, nr_bins)) for (field, feat) in feats]
    return feats

frequent_feats = read_freqent_feats(args['threshold'], isTarget = True) # features that appears more than threshold times 

with open(args['out_path'], 'w') as f:
    count = 0
    for row, line_gbdt in zip(csv.DictReader(open(args['csv_path'])), open(args['gbdt_path'])):
        if count % 1000 == 0:
        	print count
	count += 1
	feats = []

        feat_idx = 1 
        for feat in gen_feats(row):
            field = feat.split('-')[0]

            if (field in FEAT_TARGET) or (field in FEAT_CAT) and (feat not in frequent_feats):
                feat = feat.split('-')[0]+'less'

            feats.append((feat_idx, feat))
            feat_idx += 1

        if file_type == 'train':
            # skip just the first col to retrieve features
            line_feat = line_gbdt.strip().split()[1:]
        else:
            # skil the first (len[target] + 1) cols to retrieve features
            line_feat = line_gbdt.strip().split()[(len(TARGET)+1):]

        for i, feat in enumerate(line_feat, start=1):
            feats.append((feat_idx, str(i)+":"+feat))
            feat_idx += 1

        feats = gen_hashed_fm_feats(feats, args['nr_bins'])
        if file_type == 'train':
            f.write(row['Label'] + ' ' + ' '.join(feats) + '\n')
        else:
            line_w = []
            for field in TARGET:
                line_w.append(row[field])
            line_w.append(row['Label'])
            f.write(' '.join(line_w) + ' ' + ' '.join(feats) + '\n')
