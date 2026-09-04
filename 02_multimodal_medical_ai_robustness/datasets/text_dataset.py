# Prepares: input_ids, attention_mask and label

from torch.utils.data import Dataset

import torch



class ClinicalTextDataset(Dataset):


    def __init__(
            self,
            dataframe,
            tokenizer,
            max_length=256
    ):

        self.data=dataframe

        self.tokenizer=tokenizer

        self.max_length=max_length



    def __len__(self):

        return len(self.data)



    def __getitem__(
            self,
            index
    ):


        item=self.data.iloc[index]


        encoded=self.tokenizer(

            item.text_caption,

            padding="max_length",

            truncation=True,

            max_length=self.max_length,

            return_tensors="pt"

        )


        return {


            "input_ids":

                encoded["input_ids"].squeeze(0),


            "attention_mask":

                encoded["attention_mask"].squeeze(0),


            "label":

                torch.tensor(

                    item.label,

                    dtype=torch.float32

                )

        }
