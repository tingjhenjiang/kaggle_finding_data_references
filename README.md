# kaggle_finding_data_references
for kaggle "Make Data Count - Finding Data References" competitions, by team TJ, YY and JW

# project files structure

- data
    * test
        * PDF
        * XML
    * train
        * PDF
        * XML
    * sample_submission.csv
    * scientific_literatures_pdf_to_md.parquet
    * train_labels.csv
- scripts
    * hf_access_token.txt
    * preprocess_tokenize.py
    * ...

* Unzip [kaggle competition data](https://www.kaggle.com/competitions/make-data-count-finding-data-references/data) to "data" folder.
* Download scientific_literatures_pdf_to_md.parquet to "data" folder.
* Save [your huggingface access token](https://huggingface.co/settings/tokens) as "hf_access_token.txt".