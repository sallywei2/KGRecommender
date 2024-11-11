import torch

# moved from FlanT5_finetuning.ipynb
class CustomDataset(torch.utils.data.Dataset):
  def __init__(self, tokenizer, dataset, type_path, max_len=512):

    self.data = dataset
    self.max_len = max_len
    self.tokenizer = tokenizer
    self.tokenizer.max_length = max_len
    self.tokenizer.model_max_length = max_len
    self.inputs = []
    self.targets = []

    self._build()

  def __len__(self):
    return len(self.inputs)

  def __getitem__(self, index):
    source_ids = self.inputs[index]["input_ids"].squeeze()
    target_ids = self.targets[index]["input_ids"].squeeze()

    src_mask    = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
    target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

    return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

  def _preprocess_text2kgbench(self):
    # input
    #   WikiANN: ["tokens"]
    #   Text2KGBench format 1: ["sent"]
    # target
    #   WikiANN: ["spans"]
    #   Text2KGBench format 1: "sub_label": "Pent-House Mouse", "rel_label": "director", "obj_label": "Chuck Jones"
    #     We want to produce something like "director: Chuck Jones"
    
    if "rel_label" in self.data.columns and "obj_label" in self.data.columns:
        self.data["spans"] = self.data["rel_label"] + ": " + self.data["obj_label"]
    return

  def _process_spans_from_triples_text2bench(self, idx):
    #   Text2KGBench format 2: "triples": "[
    #     {'sub': '1_Decembrie_1918_University', 'rel': 'latinName', 'obj': '"Universitas Apulensis"'}, 
    #     {'sub': '1_Decembrie_1918_University', 'rel': 'country', 'obj': 'Romania'},
    #     {'sub': '1_Decembrie_1918_University', 'rel': 'state', 'obj': 'Alba'}
    #     ]"
    #     This is already provided as a single string, so we can reformat it as
    #     rel: obj; rel: obj; rel: obj...
    
    df_idx_spans = ""
    if "spans" in self.data.columns:   # Text2KGBench format 1
        df_idx_spans = "; ".join(self.data.iloc[[idx]]["spans"])
    if "triples" in self.data.columns: # Text2KGBench format 2
        spans = []
        triples = self.data.iloc[[idx]]["triples"].astype("string") # get string
        triples_dict = eval(triples[idx]) # get dict
        for i in triples_dict:
            spans.append(i["rel"] + ": " + i["obj"])
        df_idx_spans = ", ".join(spans)
    return df_idx_spans

  def _build(self):
    # pre-process dataset to fit expectations
    self._preprocess_text2kgbench()
    
    for idx in range(len(self.data)):

      # set input and target
      # input_, target = " ".join(self.data[idx]["sent"]), "; ".join(self.data[idx]["spans"])
      input_ = " ".join(self.data.iloc[[idx]]["sent"])
      target = self._process_spans_from_triples_text2bench(idx)

      input_ = input_.lower() + ' </s>'
      target = target.lower() + " </s>"

       # tokenize inputs
      tokenized_inputs = self.tokenizer.batch_encode_plus(
          [input_], max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"
      )
       # tokenize targets
      tokenized_targets = self.tokenizer.batch_encode_plus(
          [target],max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"
      )

      self.inputs.append(tokenized_inputs)
      self.targets.append(tokenized_targets)