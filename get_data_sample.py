from dataset import *
from main import *

if __name__ == "__main__":
    experiment_dataset = BioASQDataset(config)
    source_data = experiment_dataset.data_row_by_id(config.sample_idx)
    print("================= original data - question =================")
    print(str(source_data["question"]))
    print("================= original data - passage ==================")
    print("\n-".join(source_data["reference_documents"]))
