from openai import OpenAI
import math
import os

client = OpenAI(api_key=os.environ["OPENAI_PRIVATE_API_KEY"])

model = "gpt-4"

def to_prob(logprob):
  return math.exp(logprob)

def predict(s):
  response = client.chat.completions.create(
    model=model,
    messages=[    
      {"role": "user", "content": s}
    ],
    logprobs=True,
    max_tokens=1,
    top_logprobs=20,
    temperature=1
  )
  return response.choices[0].logprobs.content[0].top_logprobs

veta = "Umělá"
d = predict("Doplň mi další slovo v následující větě: " + veta)
a = predict(f"Předpokladej, že jsi ortoped, doplň mi další slovo v následující větě: {veta}")
b = predict(f"Předpokladej, že jsi kadeřnice, doplň mi další slovo v následující větě: {veta}")

print(a,b,c)

f = open("logprobs.txt", "w", encoding="utf-8")

f.write("Výchozí\n")
for token in d:
    f.write(f"{token.token}\t{to_prob(token.logprob)}\n")

f.write("Ortoped\n")
for token in a:
    f.write(f"{token.token}\t{to_prob(token.logprob)}\n")

f.write("\nKadeřnice\n")
for token in b:
    f.write(f"{token.token}\t{to_prob(token.logprob)}\n")

f.close()