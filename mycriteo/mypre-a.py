#!/usr/bin/env python3

import argparse, csv, sys

from common import *

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file_type', type=str, default='train')
parser.add_argument('-i', '--prod_num', type=str, default=0)
parser.add_argument('csv_path', type=str)
parser.add_argument('dense_path', type=str)
parser.add_argument('sparse_path', type=str)
args = vars(parser.parse_args())

#These features are dense enough (they appear in the dataset more than 4 million times), so we include them in GBDT
#target_cat_feats = ['C9-a73ee510', 'C22-', 'C17-e5ba7672', 'C26-', 'C23-32c7478e', 'C6-7e0ccccf', 'C14-b28479f6', 'C19-21ddcdc9', 'C14-07d13a8f', 'C10-3b08e48b', 'C6-fbad5c96', 'C23-3a171ecb', 'C20-b1252a9d', 'C20-5840adea', 'C6-fe6b92e5', 'C20-a458ea53', 'C14-1adce6ef', 'C25-001f3601', 'C22-ad3062eb', 'C17-07c540c4', 'C6-', 'C23-423fab69', 'C17-d4bb7bd8', 'C2-38a947a1', 'C25-e8b83407', 'C9-7cc72ec2']
target_cat_feats = read_dense_feats(threshold = 2000, isTarget = True) 

with open(args['dense_path'], 'w') as f_d, open(args['sparse_path'], 'w') as f_s:
    count = 0
    count_pos = 0
    count_neg = 0
    count_total = 0
    for row in csv.DictReader(open(args['csv_path'])):
        if count % 1000 == 0:
            print "read ", count
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

        # numerical features write into dense file
        feats = []
        for j in range(len(FEAT_NUM)):
            field = FEAT_NUM[j]
            val = row[FEAT_NUM[j]]
            #feats.append('{0}'.format(val))
            feats.append(str(val))
                
        f_d.write(str(label) + ' ' + ' '.join(feats) + '\n')
        
        # categorical features write into sparse file
        cat_feats = set()
        for field in FEAT_CAT + FEAT_TARGET:
            key = field + '-' + row[field]
            cat_feats.add(key)

        feats = []
        for j, feat in enumerate(target_cat_feats, start=1):
            if feat in cat_feats:
                feats.append(str(j))

        f_s.write(str(label) + ' ' + ' '.join(feats) + '\n')
    
    print "count_pos = ", count_pos
    print "count_neg = ", count_neg
    print "count_total = ", count_total





