from fastapi import FastAPI 
import uvicorn

from config import config

app = FastAPI()
model = transformers.BertModel.from_pretrained(
    config.BERT_PATH
)

@app.get("/sentiment_pred")
def sentiment(sentence):
    inputs = config.TOKENIZER.encode_plus(
            review,
            None,
            max_length = self.max_length,
            padding= "max_length",
            truncation= True
        )
    id = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    output = model(
        id,
        mask,
        token_type_ids
    )

    prediction = np.array(output) >0.5
    print(prediction)
    return prediction