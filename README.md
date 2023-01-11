# SemMedDB Predication Parser

## Source Data Files

There are four files required by this parser:

1. The _PREDICATION_ CSV file. 
    - Version: `semmedVER43_R`
    - Download page: https://lhncbc.nlm.nih.gov/ii/tools/SemRep_SemMedDB_SKR/SemMedDB_download.html
    - Direct download link: https://data.lhncbc.nlm.nih.gov/umls-restricted/ii/tools/SemRep_SemMedDB_SKR/semmedVER43_2022_R_PREDICATION.csv.gz
    - Filename: `semmedVER43_2022_R_PREDICATION.csv.gz`
2. The _Semantic Type Mappings_ file.
    - Version: `2018AB` (UMLS release version number, as shown in [UMLS Release File Archives](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsarchives04.html))
    - Download page: https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/documentation/SemanticTypesAndGroups.html
    - Direct download link: https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/Docs/SemanticTypes_2018AB.txt
    - Filenname: `SemanticTypes_2018AB.txt`
3. The _Retired CUI Mapping_ file.
    - Version: `2022AA` (UMLS release version number, as shown in [UMLS Release File Archives](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsarchives04.html))
    - Download page: this file is part of the [_2022AA UMLS Metathesaurus Full Subset_](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsarchives04.html); no direct download link is available.
    - Description: https://www.ncbi.nlm.nih.gov/books/NBK9685/table/ch03.T.retired_cui_mapping_file_mrcui_rr/
    - Filename: `MRCUI.RRF`
4. The _Preferred CUI Names & Semtypes_ file.
    - Github Repo: https://github.com/erikyao/UMLS_CUI_Semtype
    - How to Generate:
        - Source data files:
            - The _Retired CUI Mapping_ file, `MRCUI.RRF` (see above).
            - The _Concept Names and Sources_ file, `MRCONSO.RRF`.
                - Version: `2022AA`
                - Download page: https://www.nlm.nih.gov/research/umls/licensedcontent/umlsarchives04.html
                - Description: https://www.ncbi.nlm.nih.gov/books/NBK9685/table/ch03.T.concept_names_and_sources_file_mr/
            - The _Semantic Type Mappings_ file, `SemanticTypes_2018AB.txt` (see above).
        - Script: [parser.py](https://github.com/erikyao/UMLS_CUI_Semtype/blob/main/parser.py)
    - Filename: `UMLS_CUI_Semtype.tsv`, as shown [here](https://github.com/erikyao/UMLS_CUI_Semtype/blob/main/parser.py#L188)