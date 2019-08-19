# -*- coding: utf-8 -*-
# @m.haoran.cq@gmail.com

"""
@input: Train_labels.csv Train_review.csv
@output: Data_for_Categories_Polarities.tsv

notice:
1.

"""
polarities = {'正面': 'PP', '负面': 'PN','中性':'PU'}
categories = {'包装': 'CP', '成分': 'CC', '尺寸': 'CM', '服务': 'CS',
              '功效':'CF','价格':'CI','气味':'CT','使用体验':'CE',
              '物流':'CL','新鲜度':'CR','真伪':'CA','整体':'CG','其他':'CO'}
assert len(polarities.keys()) == len(set(polarities.values()))
assert len(categories.keys()) == len(set(categories.values()))

import pandas as pd
import numpy as np
import os

def _convert_2_int(x:str):
    try:
        return int(x)
    except:
        return x
def _append_rows(file_obj,col1:list,col2:list):
    assert len(col1)==len(col2)
    for ix,word in enumerate(col1):
        file_obj.write(word+"\t"+col2[ix]+os.linesep)
df_labels = pd.read_csv(os.path.join(os.getcwd(),'Train_labels.csv'))
df_text = pd.read_csv(os.path.join(os.getcwd(),'Train_reviews.csv'))
cat_file = open("categories.txt",mode='w')
pol_file = open("polarities.txt",mode='w')
for type in ['categories','polarities']:
    for line_ix in range(0,df_text.shape[0]):
        id = line_ix + 1
        token_list = list(df_text.iloc[line_ix,1])
        tags = ['O']*len(token_list)
        temp_df_by_id = df_labels.loc[df_labels['id']==id]
        for each_ix in range(0,temp_df_by_id.shape[0]):
            A_start = _convert_2_int(temp_df_by_id.iloc[each_ix,2])
            A_end = _convert_2_int(temp_df_by_id.iloc[each_ix,3])
            O_start = _convert_2_int(temp_df_by_id.iloc[each_ix,5])
            O_end = _convert_2_int(temp_df_by_id.iloc[each_ix,6])
            categories_value = str(temp_df_by_id.iloc[each_ix,7])
            polarities_value = str(temp_df_by_id.iloc[each_ix,8])
            if type == 'categories':
                # case1: Aspect is not None and polarities is not None
                if A_start != ' ' and O_start != ' ':
                    tags[A_start] = "B-"+categories[categories_value]
                    tags[A_start+1:A_end] = ["I-"+categories[categories_value]]*len(tags[A_start+1:A_end])
                # case2: A is None but O is not None
                if A_start == ' ' and O_start != ' ':
                    tags[O_start] = "B-"+categories[categories_value]
                    tags[O_start+1:O_end] = ["I-"+categories[categories_value]]*len(tags[O_start+1:O_end])
                _append_rows(cat_file,token_list,tags)
                cat_file.write(os.linesep)
                # case3: A is not None but O is None
            elif type == 'polarities':
                # case1: Aspect is not None and polarities is not None
                if A_start != ' ' and O_start != ' ':
                    tags[O_start] = "B-"+polarities[polarities_value]
                    tags[O_start+1:O_end] = ["I-"+polarities[polarities_value]]*len(tags[O_start+1:O_end])
                # case2: A is None but O is not None
                # case3: A is not None but O is None
                if A_start != ' ' and O_start == ' ':
                    tags[A_start] = "B-"+polarities[polarities_value]
                    tags[A_start+1:A_end] = ["I-"+polarities[polarities_value]]*len(tags[A_start+1:A_end])
                _append_rows(pol_file,token_list,tags)
                pol_file.write(os.linesep)
cat_file.close()
pol_file.close()
