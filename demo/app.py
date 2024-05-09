from flask import Flask, request, render_template
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import torch

app = Flask(__name__)

model = RobertaForSequenceClassification.from_pretrained("aalexzhang/Flair-It-RoBERTa")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

label_mapping = {
    0: "academics",
    1: "admissions",
    2: "advice",
    3: "discussion",
    4: "event",
    5: "meme",
    6: "poll",
    7: "psa",
    8: "rant",
    9: "student life"
}

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_label = ""
    if request.method == 'POST':
        title = request.form.get('title') or ""
        post = request.form.get('post') or ""
        print(title, post)
        if title.strip() == "" and post.strip() == "":
            predicted_label = ""
        else:
            inputs = tokenizer(title + " " + post, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1).item()
            predicted_label = label_mapping[predicted_class]
    return render_template('demo.html', label=predicted_label)


if __name__ == '__main__':
    app.run(debug=True)