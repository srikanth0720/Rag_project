from transformers import AutoTokenizer, RagRetriever, RagSequenceForGeneration, RagTokenForGeneration
import torch

tokenizer = AutoTokenizer.from_pretrained("facebook/rag-sequence-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="exact", use_dummy_dataset=True)
model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)


inputs = tokenizer("How many people live in Paris?", return_tensors="pt")
targets = tokenizer(text_target="In Paris, there are 10 million people.", return_tensors="pt")
input_ids = inputs["input_ids"]
labels = targets["input_ids"]

'''
outputs = model(input_ids=input_ids, labels=labels)
generated_string = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(outputs)
'''

model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", use_dummy_dataset=True)
question_hidden_states = model.question_encoder(input_ids)[0]
docs_dict = retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors="pt")
doc_scores = torch.bmm(question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)).squeeze(1)

generated = model.generate(context_input_ids=docs_dict["context_input_ids"], context_attention_mask=docs_dict["context_attention_mask"], doc_scores=doc_scores)
generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)
print(generated_string)
