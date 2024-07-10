from transformers import T5Tokenizer, T5ForConditionalGeneration

import logging

import json
import os

import torch
import pickle
import torch.nn as nn

#import argparse
#
#parser=argparse.ArgumentParser()
#parser.add_argument('--split_file',type=str,default=None)
#parser.add_argument('--model_name',type=str,default='t5-base')
#parser.add_argument('--checkpoint',type=str,default=None)
#parser.add_argument('--device',type=int,default=0)
#parser.add_argument('--beam_length',type=int,default=10)
#parser.add_argument('--save_dir',type=str,default=None)
#args=parser.parse_args()

torch.manual_seed(42)

#file=open(args.split_file,'rb')
#data=pickle.load(file)
#file.close()

#final_data_test=data[0]

class Model(nn.Module):
        def __init__(self,model_name):
                super(Model,self).__init__()
                self.model=T5ForConditionalGeneration.from_pretrained(model_name)

        def forward(self,input):
                outputs=self.model(input_ids=input['input_ids'], \
                                           labels=input['labels'],  \
                                           attention_mask=input['attention_mask'], \
                                           output_hidden_states=True,output_attentions=True)

                return outputs.loss



# TODO: get logger from module
c_handler = logging.StreamHandler()
c_handler.setLevel(logging.INFO)
c_formatter = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] %(message)s')
c_handler.setFormatter(c_formatter)

logger = logging.getLogger(__name__)
logger.addHandler(c_handler)


# args
num_gpus = 1
eval_bs = 8
beam = 10
model_name = 't5-base'
device = 0
save_dir = '.'
checkpoint = 'split_mix1_checkpoint11000.pth'

#self
tokenizer=T5Tokenizer.from_pretrained(model_name)
model=nn.DataParallel(Model(model_name),device_ids=[device])
model.to(f'cuda:{model.device_ids[0]}')

# TODO: eval.sh defines split checkpoints -> how does this work for our usecase?
# is it needed?
params=torch.load(save_dir+'/'+checkpoint, map_location='cuda:0'); # NOT self.
model.load_state_dict(params);
model.eval() # set model to evaluation mode, instead of train
logger.info("STARTED")

sparql_vocab=['?x','{','}','?uri','SELECT', 'DISTINCT', 'COUNT', '(', ')',  \
              'WHERE', '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>', \
              '.','ASK','[DEF]','<http://dbpedia.org/ontology/','<http://dbpedia.org/property/', \
              '<http://dbpedia.org/resource/']
vocab_dict={}
for i in range(len(sparql_vocab)):
    vocab_dict['<extra_id_'+str(i)+'>']=sparql_vocab[i]
vocab_dict['<extra_id_17>']=''





def preprocess_function(inputs, targets):
    model_inputs=tokenizer(inputs, padding=True, \
                                return_tensors='pt',max_length=512, truncation=True)
    labels=tokenizer(targets,padding=True,max_length=512, truncation=True)

    if True:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) \
             for l in label] for label in labels["input_ids"]
        ]
    labels['input_ids']=torch.tensor(labels['input_ids'])
    model_inputs["labels"]=labels["input_ids"].to(f'cuda:{model.device_ids[0]}')
    model_inputs["input_ids"]=model_inputs["input_ids"].to(f'cuda:{model.device_ids[0]}')
    model_inputs["attention_mask"]=model_inputs["attention_mask"].to(f'cuda:{model.device_ids[0]}')

    return model_inputs


def readable(string):
    for key in vocab_dict:
        string=string.replace(key,' '+vocab_dict[key]+' ')
    string=string.replace('  ',' ')
    vals=string.split()

    for i,val in enumerate(vals):
        if val=='<http://dbpedia.org/ontology/' or val=='<http://dbpedia.org/property/'  \
        or val=='<http://dbpedia.org/resource/':
            if i<len(vals)-1:
                vals[i]=val+vals[i+1]+'>'
                vals[i+1]=''

    return ' '.join(vals).strip().replace('  ',' ')


def answer(question: str):
    logger.info(f"attempt to answer \"{question}\"")
    bs,i = eval_bs,0
    saver = []

    # inp: text question
    # label: query
    inp,label = [],[]
    # TODO: generate inputs and labels
    # for this, check structure of test_data
    inp.append(question)
    label.append("DUMMY")

    logger.info("preprocess")
    input = preprocess_function(inp,label)

    # generate output
    logger.info("generate output")
    output=model.module.model.generate(input_ids=input['input_ids'],
                            num_beams=beam,attention_mask=input['attention_mask'], \
                            early_stopping=True, max_length=100,num_return_sequences=beam)
    out = tokenizer.batch_decode(output,skip_special_tokens=False)

    # TODO: process output
    logger.info("process ouput")
    dict={}
    # TODO: for now assuming len(inp)==1
    dict['question'] = readable(inp[0])
    dict['gold_sparql'] = readable(label[0].strip())
    dict['top_' + str(beam) + '_output'] = []
    # collect top n candidtes (n dictated by beam)
    for s in range(beam):
        dict['top_' + str(beam) + '_output']. \
            append(readable(out[beam+s].replace('<pad>','').replace('</s>','').strip()))

    logger.info(f"DICT: {dict}")
    return dict


if __name__ == '__main__':
    question = "what is the capital of Germany?"
    answer(question)



