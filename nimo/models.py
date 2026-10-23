"""Original small experimental architectures, initialized by PyTorch RNG; no weight downloads."""
import torch
from torch import nn

torch.set_num_threads(min(2, torch.get_num_threads()))


class ByteTokenizer:
    vocab_size = 259
    pad, bos, eos = 256,257,258
    def encode(self, text):
        return list(text.encode('utf-8'))
    def decode(self, ids):
        return bytes(i for i in ids if 0<=i<256).decode('utf-8',errors='replace')


class NimoLM(nn.Module):
    def __init__(self, width=64, layers=2, context=128):
        super().__init__()
        self.config = dict(width=width,layers=layers,context=context)
        self.context = context
        self.token = nn.Embedding(259,width)
        self.position = nn.Embedding(context,width)
        block = nn.TransformerEncoderLayer(width,4,width*3,dropout=0.0,batch_first=True,activation='gelu')
        self.blocks = nn.TransformerEncoder(block,layers,enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width,259,bias=False)
    def forward(self, tokens):
        length = tokens.shape[1]
        x = self.token(tokens)+self.position(torch.arange(length,device=tokens.device))
        mask = torch.triu(torch.ones(length,length,device=tokens.device,dtype=torch.bool),1)
        return self.head(self.norm(self.blocks(x,mask=mask)))
    @torch.inference_mode()
    def generate(self, prompt, limit=96):
        tok = ByteTokenizer()
        ids = [tok.bos]+tok.encode(prompt)
        output = []
        self.eval()
        for _ in range(min(limit,256)):
            x = torch.tensor([ids[-self.context:]],device=next(self.parameters()).device)
            predicted = int(self(x)[0,-1].argmax())
            if predicted in (tok.eos,tok.pad,tok.bos):
                break
            ids.append(predicted)
            output.append(predicted)
        return tok.decode(output)


class NimoVision(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.config = dict(classes=classes)
        self.features = nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((4,4)))
        self.head = nn.Sequential(nn.Flatten(),nn.Linear(512,64),nn.ReLU(),nn.Linear(64,classes))
    def forward(self,x):
        return self.head(self.features(x))


def load_model(path):
    # Local managed files only; safe tensor unpickler prevents arbitrary pickle execution.
    checkpoint = torch.load(path,map_location='cpu',weights_only=True)
    model = NimoLM(**checkpoint['architecture']) if checkpoint['task']=='text' else NimoVision(**checkpoint['architecture'])
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, checkpoint
