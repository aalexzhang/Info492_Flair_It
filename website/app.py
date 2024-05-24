from flask import Flask, request, render_template, redirect, url_for, flash
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from flask_sqlalchemy import SQLAlchemy
import torch
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
# Set secret key 
app.secret_key = os.getenv('SECRET_KEY')
db_uri = "udub.db"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + db_uri

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

color_mapping = {
    "academics": "red",
    "admissions": "blue",
    "advice": "green",
    "discussion": "purple",
    "event": "orange",
    "meme": "pink",
    "poll": "brown",
    "psa": "cyan",
    "rant": "magenta",
    "student life": "gold"
}

# Get and post route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        title = request.form.get('title') or ""
        post = request.form.get('post') or ""
        if title.strip() == "" and post.strip() == "":
            predicted_labels = ""
            flash("Error: Please enter a title or post")
        else:
            try:
                inputs = tokenizer(title + " " + post, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                print(probs)
                threshold = 0.3
                high_prob_indices = torch.where(probs > threshold)[1]
                print(high_prob_indices)
                predicted_labels = [label_mapping[index.item()] for index in high_prob_indices]
                print(predicted_labels)
                if not predicted_labels:
                    max_prob_index = torch.argmax(probs)
                    predicted_labels.append(label_mapping[max_prob_index.item()])

                predicted_labels_string = ', '.join(predicted_labels)

                new_post = Post(title=title, post=post, label=predicted_labels_string)
                db.session.add(new_post)
                db.session.commit()
                flash("Post added successfully")
            except RuntimeError as e:
                print(e)
                if "The expanded size of the tensor" in str(e):
                    flash("Error: Your text is too long. Please shorten it and try again.")
                else:
                    flash("Error: Something went wrong. Please try again.")
        return redirect(url_for('index'))

    posts = Post.query.all()
    return render_template('demo.html', posts=posts, color_mapping=color_mapping)


# Delete post button route
@app.route('/delete_post/<int:post_id>', methods=['POST'])
def delete_post(post_id):
    post_to_delete = Post.query.get_or_404(post_id)
    db.session.delete(post_to_delete)
    db.session.commit()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)