# %%
import pandas as pd
import dask.dataframe as dd
import pathlib
from pathlib import Path
import json
from typing import Optional, Union, Dict, Any, List, Tuple, Callable, Literal
import torch
import transformers
import datasets
from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer
# %%
access_token = ( (Path(__file__).parent) /"hf_access_token.txt").read_text(encoding="utf-8")

# %%
class Get_tokenized_dataset():
    def __init__(self, project_folder=Path(__file__).parent.parent):
        project_data_folder = project_folder / "data"
        self.project_data_folder = project_data_folder
        self.train_data_folder = project_data_folder/"train"
        self.test_data_folder = project_data_folder/"test"
        self.literatures_file = project_data_folder/"scientific_literatures_pdf_to_md.parquet"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cpu_device = torch.device("cpu")

    def list_all_files(self, directory:str)->List[pathlib.PosixPath]:
        path = Path(directory)
        return [file for file in path.rglob('*') if file.is_file()]

    def get_all_files_list(self, target:Literal['train','test','both']="both"):
        match target:
            case 'train':
                data_paths = self.list_all_files(self.train_data_folder)
            case 'test':
                data_paths = self.list_all_files(self.test_data_folder)
            case 'both':
                data_paths = self.list_all_files(self.train_data_folder)+self.list_all_files(self.test_data_folder)
        return data_paths

    def get_tabular_datadf_x(self)->pd.DataFrame:
        literatures = pd.read_parquet(self.literatures_file)
        data_paths = self.get_all_files_list("both")
        data_df_x = pd.DataFrame(
                {
                    'path': [str(n) for n in data_paths],
                    'filename_no_ext':[n.stem for n in data_paths],
                    'filename':[n.name for n in data_paths],
                    'extension':[n.suffix for n in data_paths],
                }
            ). \
            merge(literatures. \
                    drop(columns=['path','subset']). \
                    drop_duplicates(subset=['filename']),
                    how='left'
            ). \
            assign(subset=lambda x: (x['path'].str.find('test')==-1). \
                            replace({True:'train',False:'test'})
            )
        return data_df_x

    def get_tabular_x_and_ycompacted(self)->pd.DataFrame:
        train_labels = pd.read_csv(self.project_data_folder/"train_labels.csv")
        dev_train_labels = train_labels.drop_duplicates(subset=['article_id','dataset_id','type'])
        dev_train_labels['jsonlabel'] = dev_train_labels.apply(
            lambda x: json.dumps({
                "dataset_id":x['dataset_id'],
                'citation_type':x['type']
                }), axis=1)
        dev_train_labels = dev_train_labels.groupby('article_id')['jsonlabel'].agg(', '.join).reset_index(drop=False)
        dev_train_labels['jsonlabel'] = dev_train_labels['jsonlabel'].apply(lambda x: f"[{x}]")
        data_df_x = self.get_tabular_datadf_x()
        data_df_x_y = dev_train_labels.merge(
            data_df_x. \
                drop(columns=['path']). \
                rename(columns={'filename_no_ext':'article_id'}),
            how='left', on=['article_id'])
        return data_df_x_y

    def get_datasets_from_tabulards_unbatched(self, src:Literal['purex','x','xy']='xy', train_val_split:bool=False):
        """
        功能：將pandas dataframe格式的資料集轉為 datasets 套件格式（ https://huggingface.co/docs/datasets/package_reference/main_classes ）的資料集
        src：選擇資料集範圍。競賽主辦單位給的資料分有trainset、testset。
            根據比賽方法1（純prompt engineering）比賽方法2（訓練gen llm）比賽方法3（token classification）需求範圍不同，因此區別轉換範圍
            其中trainset實際上僅有部分資料有標籤；
            本來就沒有標籤而用來作為競賽送出答案的預測資料集testset筆數很少；
            purex資料量最完全，包含所有trainset資料和testset資料，不包含標籤；
            x則是來自於「有標籤的資料」，但去除標籤
            xy則是來自於「有標籤的資料」，且不去除標籤
        """
        match src:
            case 'purex':
                tab_df = self.get_tabular_datadf_x()
            case 'x' | 'xy':
                tab_df = self.get_tabular_x_and_ycompacted()
        hf_ds = datasets.Dataset.from_pandas(tab_df.loc[tab_df['subset']=='train',:], preserve_index=False)
        hf_ds_testset = datasets.Dataset.from_pandas(tab_df.loc[tab_df['subset']=='test',:], preserve_index=False)
        if train_val_split:
            hf_ds = hf_ds.train_test_split(test_size=0.2, shuffle=True)
            hf_ds["validation"] = hf_ds.pop("test")
            hf_ds['test'] = hf_ds_testset
        else:
            hf_ds = datasets.DatasetDict({
                "train": hf_ds,
                "test": hf_ds_testset
            })
        hf_ds = datasets.DatasetDict(hf_ds)
        return hf_ds

    def init_tokenizer(self, model_id:str = "google/gemma-3-1b-it"):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id,
            token=access_token)

    def add_instructions_on_datasets(
            inputdatasets:datasets.dataset_dict.DatasetDict,
            systemprompt:str="",
            instruct:str=""
            )->datasets.dataset_dict.DatasetDict:
        """
        運用 tokenizer 替原始資料集加上instruction、對話以便和LLM對話，prompt engineering
        """
        pass

    def do_tokenize(self, hf_ds, model_id:str = "google/gemma-3-1b-it"):
        """
        將 datasets 套件格式的文字資料集運用tokenizer轉換為可以餵進模型的一批一批tokens，有分批功能，並且在同一批資料內padding補齊至同樣長度的功能等
        """
        if getattr(self, "model", None) is None:
            self.init_tokenizer(model_id=model_id)
        # ...

# %%
if __name__=='__main__':
    get_generation_dataset_instance = Get_tokenized_dataset()
    # tabdf = get_generation_dataset_instance.get_tabular_x_and_ycompacted()
    tabdf = get_generation_dataset_instance.get_datasets_from_tabulards_unbatched(
        src='xy',
        train_val_split=True
        )
    print(tabdf)