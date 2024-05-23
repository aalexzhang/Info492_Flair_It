from flask import Flask, request, render_template, redirect, url_for
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from flask_sqlalchemy import SQLAlchemy
import torch

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///udub.db'

db = SQLAlchemy(app)
class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(120), nullable=False)
    post = db.Column(db.String(120), nullable=False)
    label = db.Column(db.String(120), nullable=False)

with app.app_context():
    db.create_all()

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
    if request.method == 'POST':
        title = request.form.get('title') or ""
        post = request.form.get('post') or ""
        print(title, post)
        if title.strip() == "" and post.strip() == "":
            predicted_labels = ""
        else:
            try:
                inputs = tokenizer(title + " " + post, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                print(probs)
                threshold = 0.3
                high_prob_indices = torch.where(probs > threshold)[1]
                predicted_labels = [label_mapping[index.item()] for index in high_prob_indices]
                predicted_labels_string = ', '.join(predicted_labels)

                new_post = Post(title=title, post=post, label=predicted_labels_string)
                db.session.add(new_post)
                db.session.commit()
            except RuntimeError as e:
                if "Error: Your text is too long. Please shorten it and try again." in str(e):
                    return "Error: Your text is too long. Please shorten it and try again."
        return redirect(url_for('index'))

    posts = Post.query.all()
    return render_template('demo.html', posts=posts)


@app.route('/delete_post/<int:post_id>', methods=['POST'])
def delete_post(post_id):
    post_to_delete = Post.query.get_or_404(post_id)
    db.session.delete(post_to_delete)
    db.session.commit()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)