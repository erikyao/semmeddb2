# SemMedDB Predication Parser

## Source Data Files

There are four files required by this parser:

1. `semmedVER43_2022_R_PREDICATION.csv.gz`, the _PREDICATION_ CSV file
    - Version: `43`
    - [Download page](https://lhncbc.nlm.nih.gov/ii/tools/SemRep_SemMedDB_SKR/SemMedDB_download.html) (**UMLS login required**)
2. `SemanticTypes_2018AB.txt`, the _Semantic Type Mappings_ file
    - Version: `2018AB`
    - [Download page](https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/documentation/SemanticTypesAndGroups.html)
3. `MRCUI.RRF`, the _Retired CUI Mapping_ file
    - Version: `2022AA`
    - Download page: this file is part of the [2022AA UMLS Metathesaurus Full Subset](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsarchives04.html) (**UMLS login required**); no direct download link is available.
    - [Description page](https://www.ncbi.nlm.nih.gov/books/NBK9685/table/ch03.T.retired_cui_mapping_file_mrcui_rr/)
4. `UMLS_CUI_Semtype.tsv`, the _Preferred CUI Names & Semtypes_ file.
    - Github Repo: https://github.com/erikyao/UMLS_CUI_Semtype
    - How to Generate:
        - Source data files:
            1. `MRCUI.RRF` (see above)
            2. `MRCONSO.RRF`, the _Concept Names and Sources_ file
                - Version: `2022AA`
                - [Download page](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsarchives04.html) (**UMLS login required**)
                - [Description page](https://www.ncbi.nlm.nih.gov/books/NBK9685/table/ch03.T.concept_names_and_sources_file_mr/)
            3. `MRSTY.RRF`, the _Semantic Types_ file
                - Version: `2022AA`
                - Download page: this file is part of the [2022AA UMLS Metathesaurus Full Subset](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsarchives04.html) (**UMLS login required**); no direct download link is available.
                - [Description page](https://www.ncbi.nlm.nih.gov/books/NBK9685/table/ch03.Tf/)
            4. `SemanticTypes_2018AB.txt` (see above)
        - Script: [parser.py](https://github.com/erikyao/UMLS_CUI_Semtype/blob/main/parser.py)
            - Filename is as hardcoded [here](https://github.com/erikyao/UMLS_CUI_Semtype/blob/main/parser.py#L188).