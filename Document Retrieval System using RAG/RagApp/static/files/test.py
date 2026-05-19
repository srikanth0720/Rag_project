from transformers import RagTokenizer, RagSequenceForGeneration

# Replace with your chosen model and tokenizer names
model_name = "facebook/rag-token-base"
tokenizer = RagTokenizer.from_pretrained(model_name)
model = RagSequenceForGeneration.from_pretrained(model_name)

inputs = tokenizer('What is Task Decomposition?', return_tensors="pt")
#print(dir(RagSequenceForGeneration))
#retrieval_output = model.get_retrieval_vector(inputs)

generation_inputs = {
    "input_ids": inputs.input_ids,
    "attention_mask": inputs.attention_mask
    
}
generation_output = model.generate(**generation_inputs)
generated_text = tokenizer.decode(generation_output.sequences[0])

#print(f"Retrieved documents:", retrieval_output)
print(f"Generated text:", generated_text)
