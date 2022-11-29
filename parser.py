import csv
import os
import pandas as pd
import re


def construct_rec(line, type_label):
    """
    SemMedDB Database Details: https://lhncbc.nlm.nih.gov/ii/tools/SemRep_SemMedDB_SKR/dbinfo.html

    Name: PREDICATION table
    Each record in this table identifies a unique predication. The data fields are as follows:

    PREDICATION_ID  : Auto-generated primary key for each unique predication
    SENTENCE_ID     : Foreign key to the SENTENCE table
    PMID            : The PubMed identifier of the citation to which the predication belongs
    PREDICATE       : The string representation of each predicate (for example TREATS, PROCESS_OF)
    SUBJECT_CUI     : The CUI of the subject of the predication
    SUBJECT_NAME    : The preferred name of the subject of the predication
    SUBJECT_SEMTYPE : The semantic type of the subject of the predication
    SUBJECT_NOVELTY : The novelty of the subject of the predication
    OBJECT_CUI      : The CUI of the object of the predication
    OBJECT_NAME     : The preferred name of the object of the predication
    OBJECT_SEMTYPE  : The semantic type of the object of the predication
    OBJECT_NOVELTY  : The novelty of the object of the predication
    """
    reg = re.compile('^[C0-9|\.]+$')  # CUI format check

    # skip some error records (e.g., predication id:80264847,123980473).
    if reg.match(line[8].strip()):
        predication_id = line[0]
        # ignore line[1]
        pmid = int(line[2])
        pred = line[3]

        sub_umls = line[4].split("|")
        sub_name = line[5].split("|")
        sub_semtype = line[6]
        sub_novelty = int(line[7])

        obj_umls = line[8].split("|")
        obj_name = line[9].split("|")
        obj_semtype = line[10]
        obj_novelty = int(line[11])

        # Find UMLS mapping
        sub_semtype_name = type_label.get(sub_semtype, None)
        obj_semtype_name = type_label.get(obj_semtype, None)

        sub_id_field = "umls"
        obj_id_field = "umls"

        # Define ID field name
        if "C" not in line[4]:  # one or more gene ids
            sub_id_field = "ncbigene"
        else:
            if '|' in line[4]:
                # take first CUI if it contains gene id(s)
                sub_umls = [sub_umls[0]]
                sub_name = [sub_name[0]]

        if "C" not in line[8]:  # one or more gene ids
            obj_id_field = "ncbigene"
        else:
            if '|' in line[8]:  # take first CUI if it contains gene id(s)
                obj_umls = [obj_umls[0]]
                obj_name = [obj_name[0]]

        rec_dict_list = []
        id_count = 0  # loop to get all id combinations if one record has multiple ids
        for sub_idx, sub_id in enumerate(sub_umls):
            for obj_idx, obj_id in enumerate(obj_umls):

                id_count += 1
                if len(sub_umls) == 1 and len(obj_umls) == 1:
                    id_value = predication_id
                else:
                    id_value = predication_id + "_" + str(id_count)  # add sequence id

                rec_dict = {
                    "_id": id_value,
                    "predicate": pred,
                    "predication_id": predication_id,
                    "pmid": pmid,
                    "subject": {
                        sub_id_field: sub_id,
                        "name": sub_name[sub_idx],
                        "semantic_type_abbreviation": sub_semtype,
                        "semantic_type_name": sub_semtype_name,
                        "novelty": sub_novelty
                    },
                    "object": {
                        obj_id_field: obj_id,
                        "name": obj_name[obj_idx],
                        "semantic_type_abbreviation": obj_semtype,
                        "semantic_type_name": obj_semtype_name,
                        "novelty": obj_novelty
                    }
                }

                # del semtype_name field if we did not any mappings
                if not sub_semtype_name:
                    del rec_dict["subject"]["semantic_type_name"]
                if not obj_semtype_name:
                    del rec_dict["object"]["semantic_type_name"]
                rec_dict_list.append(rec_dict)

        return rec_dict_list


def load_data(data_folder):
    edges_path = os.path.join(data_folder, "semmed_0821.csv")
    mapping_path = os.path.join(data_folder, "SemanticTypes_2013AA.txt")
    names = pd.read_csv(mapping_path, sep="|", names=['abv', 'ID', 'label'])
    type_label = dict(zip(names.abv, names.label))

    with open(edges_path) as f:  # get total record count
        next(f)
        csv_total = sum(1 for line in f)

    with open(edges_path) as f:  # data prep
        csv_reader = csv.reader(f, delimiter=';')
        next(csv_reader)
        count = 0
        for _item in csv_reader:
            count += 1
            print("Data Generation Progess:", str(count)+"/"+str(csv_total))
            records = construct_rec(_item, type_label)
            if records:
                for record in records:
                    yield record
            print("=====")
        print("Data Generation is Done.")
