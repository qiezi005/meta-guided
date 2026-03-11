import torch
import tqdm

from openai import OpenAI


def generate_embeddings(path, model='gpt', datatoken=''):
   
    OPENAI_API_KEY = "xxxx"  # TODO: change API key
    client = OpenAI(api_key=OPENAI_API_KEY)
    embedding_model = "text-embedding-3-large"   # 或 "text-embedding-3-small"
    device = torch.device("cpu")
    lm = None  
    
    f = open(path, 'r')
    embeddings = []
    for line in tqdm.tqdm(f.readlines()):
        resp = client.embeddings.create(
        model=embedding_model,
        input=line
        )    
        vec = torch.tensor(resp.data[0].embedding, dtype=torch.float32)
        embeddings.append(vec)
        torch.save(embeddings, path[:-4] + "_gpt.pt")
        
    f.close()


# generate_embeddings("data/prompts/ROI_prompts/AAL116_short.txt", datatoken='ROI')
generate_embeddings("/Users/xiajing/Library/CloudStorage/OneDrive-Personal/code/meta-prompt/MT/language_models/Subject_prompt_SZ_SRPBS.txt", datatoken='meta')
#generate_embeddings("/root/autodl-tmp/data/meta-prompt/MT/language_models/Subject_prompt2.txt", datatoken='meta')
#generate_embeddings("/root/autodl-tmp/data/meta-prompt/MT/language_models/Subject_prompt2.txt", datatoken='meta')