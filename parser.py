from collections import defaultdict
import csv
import os
import sys
import pandas as pd
import json
import re

def load_data(data_folder):

    def construct_rec(line):

      reg=re.compile('^[C0-9|\.]+$') ### CUI format check

      if(reg.match(line[8].strip())): ### skip some error records (e.g., predication id:80264847,123980473). 
        sub_umls=line[4].split("|")
        obj_umls=line[8].split("|")
        predication_id=line[0]
        sub_semtype=line[6]
        sub_name=line[5].split("|")
        obj_semtype=line[10]
        obj_name=line[9].split("|")
        pmid=int(line[2])
        pred=line[3]
        sub_novelty=int(line[7])
        obj_novelty=int(line[11])
        sub_id_field="umls"
        obj_id_field="umls"
 
        sub_semtype_name=None
        obj_semtype_name=None

        ###Find UMLS mapping
        if (sub_semtype in type_label):
         sub_semtype_name=type_label[sub_semtype]
       
        if (obj_semtype in type_label):
         obj_semtype_name=type_label[obj_semtype]
       
        ### Define ID field name
        if ("C" not in line[4]): ### one or more gene ids
           sub_id_field="ncbigene"         
        else:
           if ('|' in line[4]):
            sub_umls=[sub_umls[0]] ### take first CUI if it contains gene id(s)
            sub_name=[sub_name[0]]

        if ("C" not in line[8]): ### one or more gene ids
           obj_id_field="ncbigene"
        else:
           if ('|' in line[8]): ### take first CUI if it contains gene id(s)
              obj_umls=[obj_umls[0]]
              obj_name=[obj_name[0]]

        rec_dict_list=[]
        id_count=0 ### loop to get all id combinations if one record has multiple ids
        for sub_idx,sub_id in enumerate(sub_umls):
         for obj_idx,obj_id in enumerate(obj_umls):

           id_count+=1
           if (len(sub_umls)==1 and len(obj_umls)==1): 
              id_value=predication_id 
           else: 
              id_value=predication_id+"_"+str(id_count) ### add sequence id


           rec_dict={
             "_id": id_value,
             "predicate": pred,
             "predication_id":predication_id,
             "pmid": pmid,
             "subject":{
                    sub_id_field: sub_id,
                    "name": sub_name[sub_idx],
                    "semantic_type_abbreviation": sub_semtype,
                    "semantic_type_name": sub_semtype_name,
                    "novelty": sub_novelty
                  },
             "object":{
                   obj_id_field: obj_id,
                   "name": obj_name[obj_idx],
                   "semantic_type_abbreviation": obj_semtype,
                   "semantic_type_name": obj_semtype_name, 
                   "novelty": obj_novelty
                  }
           }

           ### del semtype_name field if we did not any mappings
           if (not sub_semtype_name): del rec_dict["subject"]["semantic_type_name"]
           if (not obj_semtype_name): del rec_dict["object"]["semantic_type_name"]
           rec_dict_list.append(rec_dict)

        return rec_dict_list
  

    edges_path = os.path.join(data_folder, "semmed_0821.csv")
    mapping_path = os.path.join(data_folder, "SemanticTypes_2013AA.txt")
    names = pd.read_csv(mapping_path, sep="|",names=['abv', 'ID', 'label'])
    type_label = dict(zip(names.abv, names.label))

    with open(edges_path) as f: ### get total record count
       next(f)
       csv_total=sum(1 for line in f)

    
    with open(edges_path) as f: ### data prep
        csv_reader = csv.reader(f, delimiter=';')
        next(csv_reader)
        count=0
        for _item in csv_reader:
             count+=1
             print("Data Generation Progess:",str(count)+"/"+str(csv_total))
             records=construct_rec(_item)
             if(records):   
              for record in records:
                yield record
             print("=====")
        print("Data Generation is Done.")



