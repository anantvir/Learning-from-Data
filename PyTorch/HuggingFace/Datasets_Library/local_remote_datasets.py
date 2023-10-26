from datasets import load_dataset

squad_it_dataset = load_dataset("json", data_files="D:\PyTorch\Learning-from-Data\PyTorch\dataset\squad\SQuAD_it-train.json\SQuAD_it-train.json", field="data")
"""We use field="data" because data is not line separated JSON, instead its entire JSON object"""

#print(squad_it_dataset["train"][0])

data_files = {"train": "SQuAD_it-train.json", "test": "SQuAD_it-test.json"}
squad_it_dataset = load_dataset("json",data_dir="D:\PyTorch\Learning-from-Data\PyTorch\dataset\squad\combined_train_and_test", data_files=data_files, field="data")
print(squad_it_dataset)


#---------------------------------------------Load remote dataset from url-----------------------------------------------------------

url = "https://github.com/crux82/squad-it/raw/master/"
data_files = {
    "train": url + "SQuAD_it-train.json.gz",
    "test": url + "SQuAD_it-test.json.gz",
}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")

print(squad_it_dataset)



















