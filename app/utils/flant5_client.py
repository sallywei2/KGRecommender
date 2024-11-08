from google.cloud import aiplatform, aiplatform_v1
from google.api import httpbody_pb2
import pandas as pd
import json

# prompt Flan-T5
from torch.utils.data import DataLoader
from models.flant5.CustomDataset import CustomDataset
from transformers import (
    AutoTokenizer
)
import textwrap

from .rag_constants import FLANT5_ENDPOINT

class FlanT5Client:

    BATCH_SIZE = 5
    BATCH_LIMIT = 0 # set to 0 to look at everything

    def __init__(self,
                 endpoint = aiplatform.Endpoint(FLANT5_ENDPOINT)):
        self.endpoint = endpoint
        self.tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-small')

    def generate_content(self, text):
        """
        Function name matches google.vertexai.GenerativeModel
        """
        try:
            response = self._make_raw_request(text)
            decoded_response = self.decode_output(response) #.predictions[0]
            return decoded_response
        except Exception as e:
            print(f"Error generating content: {e}")
            return ""

    def _get_instance(self, text):
        """
        Reformats text in FlanT5's CustomDataset format
        """
        df = pd.DataFrame(data={'title':'', 'sent':text}, index=[0])
        dataset = CustomDataset(tokenizer=self.tokenizer, dataset=df, type_path='test')

        dataloader = DataLoader(dataset, batch_size=self.BATCH_SIZE, num_workers=1, shuffle=False)
        for batch in dataloader:
            print(batch)
            input_ids =  batch['source_ids']
            attention_mask =  batch['source_mask']
            break
        
        return input_ids, attention_mask

    def _serialize_tensor(self, tensor):
        # Convert tensor to NumPy array
        numpy_array = tensor.numpy()

        # Serialize the NumPy array as JSON
        json_string = json.dumps(numpy_array.tolist())
        return json_string

    def _make_raw_request(self, text):
        """
        make a raw HTTP request to send to the AI's endpoint
        """
        input_ids, attention_mask = self._get_instance(text)
        DATA = {
            "signature_name": "predict",
            "instances": [
                {
                    "input_ids": self._serialize_tensor(input_ids),
                    "attention_mask": self._serialize_tensor(attention_mask),
                }
            ],
        }

        http_body = httpbody_pb2.HttpBody(
            data=json.dumps(DATA).encode("utf-8"),
            content_type="application/json",
        )

        req = aiplatform_v1.RawPredictRequest(
            http_body=http_body, endpoint=self.endpoint.resource_name
        )

        pred_client = aiplatform.gapic.PredictionServiceClient(client_options=
                        {"api_endpoint": FLANT5_ENDPOINT}
                    )

        #response = self.endpoint.predict(req)
        response = pred_client.raw_predict(req)
        return response

    def decode_output(self, output):
        dec = self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        return dec
