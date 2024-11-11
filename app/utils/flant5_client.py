from google.cloud import aiplatform, aiplatform_v1
from google.api import httpbody_pb2
import pandas as pd
import json
import tensorflow as tf
import torch

# prompt Flan-T5
from torch.utils.data import DataLoader
from app.models.flant5.CustomDataset import CustomDataset
from app.models.flant5.FlanT5 import T5FineTuner
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    TFT5ForConditionalGeneration,
    AutoTokenizer
)
import textwrap

from .rag_constants import FLANT5_API_ENDPOINT, FLANT5_ENDPOINT

class FlanT5Client:

    BATCH_SIZE = 2
    BATCH_LIMIT = 0 # set to 0 to look at everything

    # local checkpoint fallback if cloud fails
    LOCAL_CKPT = "models/flant5/epoch=3-step=2072-train_loss=0.35.ckpt"

    def __init__(self, use_cloud=False, use_pretrained=False):
       self.CLOUD = use_cloud
       self.PRETRAINED = use_pretrained
       
       if self.CLOUD:
           self._init_cloud() 
       else:
           self._init_local()

    def _init_local(self):
        """
        Initialize tokenizer and model
        """
        if self.PRETRAINED:
            self.tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-small')

            checkpoint = torch.load(self.LOCAL_CKPT)
            print(f"Loaded local model from checkpoint: {checkpoint.keys()}")
            self.llm = T5FineTuner.load_from_checkpoint(self.LOCAL_CKPT)
            self.llm = self.llm.to("cpu") # use CPU since I don't have GPU

            self.model = self.llm.model
            self.model.eval() # set model to evaluation mode
        else:
            self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
            self.model = TFT5ForConditionalGeneration.from_pretrained("google/flan-t5-base")        

    def _init_cloud(self):
        """
        Initialize tokenizer and API endpoint for the model hosted in the cloud
        """
        self.tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-small')
        client_options = {"api_endpoint": FLANT5_API_ENDPOINT}
        self.client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

        self.endpoint = aiplatform.Endpoint(FLANT5_ENDPOINT)
        print(f"api_endpoint: {FLANT5_API_ENDPOINT}, endpoint: {self.endpoint}")

        # split FLANT5_ENDPOINT:
        # projects/371748443295/locations/us-west1/endpoints/8518052919822516224
        self.project = FLANT5_ENDPOINT.split('/')[1]
        self.location = FLANT5_ENDPOINT.split('/')[3]
        self.endpoint_id = FLANT5_ENDPOINT.split('/')[5]
        
        print(f"project: {self.project}, endpoint_id: {self.endpoint_id}, location: {self.location}")
        
    def generate_content(self, text):
        """
        Function name matches google.vertexai.GenerativeModel
        """
        try:
            if self.CLOUD:
                response = self._predict_from_cloud(text)
            else:
                response = self._predict_locally(text)
            decoded_response = self.decode_output(response) #.predictions[0]
            return decoded_response
        except Exception as e:
            print(f"Error generating content: {e}")
            raise e

    def _get_instance(self, text):
        """
        Reformats text in FlanT5's CustomDataset format
        """
        text = text[:512]
        df = pd.DataFrame(data={'title':['',''], 'sent':[text,'']}, index=[0,1])
        dataset = CustomDataset(tokenizer=self.tokenizer, dataset=df, type_path='test')

        dataloader = DataLoader(dataset, batch_size=self.BATCH_SIZE, num_workers=1, shuffle=False)
        for batch in dataloader:
            input_ids =  batch['source_ids']
            attention_mask =  batch['source_mask']
            break
        
        input_ids = self._serialize_tensor(input_ids)
        attention_mask = self._serialize_tensor(attention_mask)

        return input_ids, attention_mask

    def _serialize_tensor(self, tensor):
        """
        The PyTorch Lightning model expects tensorflow tensors for its inputs.
        """
        # Convert tensor to NumPy array
        numpy_array = tensor.numpy()
        #tf_tensor = tf.convert_to_tensor(numpy_array, dtype=tf.int32)

        # Serialize the NumPy array as JSON
        #json_string = json.dumps(numpy_array.tolist())
        #numpy_array#json_string
        return numpy_array

    def _predict_locally(self, text):
        input_ids, attention_mask = self._get_instance(text)
        print(input_ids)
        return self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                    )

    def _predict_from_cloud(self, text):
        """
        ref: https://github.com/googleapis/python-aiplatform/blob/main/samples/snippets/prediction_service/predict_custom_trained_model_sample.py
        
        Error generating content: 404 Endpoint `projects/371748443295/locations/us-west1/endpoints/8518052919822516224` not found.
        """
        input_ids, attention_mask = self._get_instance(text)
        instances= [
            {
            "input_ids": input_ids.tolist(), #VertexAI wants a list https://stackoverflow.com/a/75811588
            "attention_mask": attention_mask.tolist(),
            }
        ]
        
        #instances = instances if isinstance(instances, list) else [instances]
        #instances = [
        #    json_format.ParseDict(instance_dict, Value()) for instance_dict in instances
        #]
        #parameters_dict = {}
        #parameters = json_format.ParseDict(parameters_dict, Value())
        endpoint = self.client.endpoint_path(
            project=self.project, location=self.location, endpoint=self.endpoint_id
        )
        print(f"Submitting to endpoint: {endpoint}")
        response = self.client.predict(
            endpoint=endpoint, instances=instances#, parameters=parameters
        )
        print("Received response:", response)
        print("deployed_model_id:", response.deployed_model_id)
        # The predictions are a google.protobuf.Value representation of the model's predictions.
       # predictions = response.predictions
        #for prediction in predictions:
        #    print(" prediction:", prediction)

        return response

    def decode_output(self, output):
        decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        return decoded_output