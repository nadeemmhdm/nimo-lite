import pytest
import torch

from nimo.conversation import ConversationStore
from nimo.corpus import CorpusStore
from nimo.lm_training import _pretrain_sequences, _sft_pair
from nimo.models import NimoLMV2, load_model
from nimo.retrieval import bm25
from nimo.store import Store
from nimo.tokenizer import NimoTokenizer, train_tokenizer


def tokenizer_fixture():
    texts=[
        "Hello Nimo. This tokenizer learns local text.",
        "നിന്റെ പേര് എന്താണ്? എന്റെ പേര് നിമോ.",
        "Ninte peru entha? Ente project Nimo aanu.",
        "def add(a, b):\n    return a + b\n",
        "x² + 2x + 1 = (x + 1)²",
    ]*20
    return train_tokenizer(texts,vocab_size=512)


def test_native_tokenizer_roundtrip_multilingual_code_math(tmp_path):
    tok=tokenizer_fixture()
    cases=["Hello world","നിന്റെ പേര് എന്താണ്?","Ninte peru entha?","def x():\n    return 42","π ≈ 3.14159"]
    for text in cases:
        assert tok.decode(tok.encode(text))==text
    path=tmp_path/"tokenizer.json";tok.save(path)
    loaded=NimoTokenizer.load(path)
    assert loaded.vocab_size==tok.vocab_size
    assert loaded.decode(loaded.encode(cases[1]))==cases[1]


def test_causal_attention_does_not_see_future_tokens():
    torch.manual_seed(1)
    model=NimoLMV2(vocab_size=512,width=64,layers=2,heads=4,ff=128,context=64,dropout=0)
    model.eval()
    a=torch.tensor([[10,11,12,13]])
    b=torch.tensor([[10,11,99,98]])
    with torch.inference_mode():
        la=model(a);lb=model(b)
    assert torch.allclose(la[:,:2],lb[:,:2],atol=1e-5)


def test_pretrain_and_sft_targets_mask_non_assistant_tokens():
    tok=tokenizer_fixture()
    rows=[{"text":"A short local pretraining document."}]
    seq=_pretrain_sequences(rows,tok,64)
    assert seq and len(seq[0][0])==len(seq[0][1])
    pair=_sft_pair(tok,[{"role":"system","content":"You are Nimo."},{"role":"user","content":"Hello"},
                        {"role":"assistant","content":"Hello. How can I help?"}],64)
    x,y=pair
    assert len(x)==len(y)
    assert any(v==-100 for v in y) and any(v!=-100 for v in y)


def test_one_native_pretraining_step_updates_weights():
    tok=tokenizer_fixture()
    model=NimoLMV2(vocab_size=tok.vocab_size,width=64,layers=1,heads=4,ff=128,context=64)
    rows=[{"text":"Local language model training text. "*8}]
    x,y=_pretrain_sequences(rows,tok,64)[0]
    x=torch.tensor([x]);y=torch.tensor([y])
    before=model.token.weight.detach().clone()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
    loss=torch.nn.functional.cross_entropy(model(x).reshape(-1,tok.vocab_size),y.reshape(-1))
    loss.backward();opt.step()
    assert torch.isfinite(loss)
    assert not torch.equal(before,model.token.weight.detach())


def test_checkpoint_contains_tokenizer_and_safe_loads(tmp_path):
    tok=tokenizer_fixture()
    model=NimoLMV2(vocab_size=tok.vocab_size,width=128,layers=1,heads=4,ff=256,context=128)
    path=tmp_path/"model.pt"
    torch.save({"task":"text","architecture":model.config,"model":model.state_dict(),"tokenizer":tok.artifact(),
                "tokenizer_sha256":tok.artifact()["sha256"]},path)
    restored,ck=load_model(path)
    assert isinstance(restored,NimoLMV2)
    assert ck["tokenizer"]["format"]=="nimo-tokenizer"


def test_conversation_history_and_edit_delete(tmp_path):
    store=Store(tmp_path);chats=ConversationStore(store)
    chat=chats.create();mid=chats.add(chat["id"],"user","hello")["id"]
    chats.add(chat["id"],"assistant","hi")
    chats.edit_message(chat["id"],mid,"hello again")
    assert chats.messages(chat["id"])[0]["content"]=="hello again"
    chats.delete_message(chat["id"],mid)
    assert len(chats.messages(chat["id"]))==1


def test_bm25_prefers_relevant_local_knowledge():
    rows=[{"text":"Python loops repeat a block of code","id":"a"},{"text":"Bananas are fruit","id":"b"}]
    assert bm25("python loop",rows)[0]["id"]=="a"


def test_corpus_exact_near_dedup_and_archive_traversal(tmp_path):
    store=Store(tmp_path);corpus=CorpusStore(store)
    a=corpus.add("Nimo local dataset example","me","CC0",status="approved")
    b=corpus.add("Nimo local dataset example","me","CC0",status="approved")
    assert a["duplicate"] is False and b["duplicate"]=="exact"
    import io,zipfile
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,"w") as z:z.writestr("../escape.txt","bad")
    with pytest.raises(ValueError,match="traversal"):
        corpus.import_bytes("bad.zip",buf.getvalue(),"test","CC0")


def test_generation_deterministic_seed_and_kv_cache():
    tok=tokenizer_fixture()
    model=NimoLMV2(vocab_size=tok.vocab_size,width=64,layers=1,heads=4,ff=128,context=64)
    prompt=tok.encode_chat([{"role":"user","content":"Hello"}],True)
    a=list(model.generate_ids(prompt,tok,limit=8,seed=7))
    b=list(model.generate_ids(prompt,tok,limit=8,seed=7))
    assert a==b
    first=torch.tensor([prompt[:4]])
    logits,cache=model(first,use_cache=True)
    assert cache[0][0].shape[2]==len(prompt[:4])
