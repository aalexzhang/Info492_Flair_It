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
    predicted_labels = ""
    if request.method == 'POST':
        title = request.form.get('title') or ""
        post = request.form.get('post') or ""
        print(title, post)
        if title.strip() == "" and post.strip() == "":
            predicted_labels = ""
        else:
            inputs = tokenizer(title + " " + post, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            print(probs)
            threshold = 0.3
            high_prob_indices = torch.where(probs > threshold)[1]
            predicted_labels = [label_mapping[index.item()] for index in high_prob_indices]
    return render_template('demo.html', label=predicted_labels)


if __name__ == '__main__':
    app.run(debug=True)