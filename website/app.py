from flask import Flask, request, render_template, redirect, url_for, flash, session
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from flask_sqlalchemy import SQLAlchemy
import torch
from dotenv import load_dotenv
import os
import json

load_dotenv()

app = Flask(__name__)

app.secret_key = os.getenv('SECRET_KEY')

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///posts.db'

db = SQLAlchemy(app)

class Post(db.Model):
    __abstract__ = True
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(120), nullable=False)
    post = db.Column(db.String(120), nullable=False)
    label = db.Column(db.String(120), nullable=False)

class udub(Post):
    __tablename__ = 'udub'

class rutgers(Post):
    __tablename__ = 'rutgers'

class usc(Post):
    __tablename__ = 'usc'

class nyu(Post):
    __tablename__ = 'nyu'

class uiuc(Post):
    __tablename__ = 'uiuc'

with app.app_context():
    db.create_all()

model = RobertaForSequenceClassification.from_pretrained("aalexzhang/Flair-It-RoBERTa-udub")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'post_model' not in session:
        return render_template('start.html')
    post_model = globals()[session['post_model']]
    model_name = session.get('model')
    if model_name:
        model = RobertaForSequenceClassification.from_pretrained(model_name)
    label_mapping = session.get('label_mapping')
    color_mapping = session.get('color_mapping')
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
                threshold = 0.3
                high_prob_indices = torch.where(probs > threshold)[1]
                print(label_mapping)
                label_mapping = {int(k): v for k, v in label_mapping.items()}
                predicted_labels = [label_mapping[index.item()] for index in high_prob_indices]
                if not predicted_labels:
                    max_prob_index = torch.argmax(probs)
                    predicted_labels.append(label_mapping[max_prob_index.item()])

                predicted_labels_string = ', '.join(predicted_labels)

                new_post = post_model(title=title, post=post, label=predicted_labels_string)
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
    posts = post_model.query.all()
    print(posts, "SDFSDFSDFSDFSDFSDF")
    return render_template('demo.html', posts=posts, color_mapping=color_mapping, filtered=False)

@app.route('/<selection>', methods=['GET'])
def select(selection):
    with open('config.json') as f:
        configs = json.load(f)
    config = configs.get(selection)
    if config:
        session['model'] = config['model']
        session['label_mapping'] = {int(k): v for k, v in config['label_mapping'].items()}
        session['color_mapping'] = config['color_mapping']
        session['post_model'] = config['post_model']
    return redirect(url_for('index'))

@app.route('/filter/<label>', methods=['GET'])
def filter(label):
    post_model = globals()[session['post_model']]
    posts = post_model.query.filter(post_model.label.contains(label)).all()
    color_mapping = session.get('color_mapping')
    return render_template('demo.html', posts=posts, color_mapping=color_mapping, filtered=True)

@app.route('/unfilter', methods=['GET'])
def unfilter():
    return redirect(url_for('index'))

@app.route('/delete_post/<int:post_id>', methods=['POST'])
def delete_post(post_id):
    post_model = globals()[session['post_model']]
    post_to_delete = post_model.query.get_or_404(post_id)
    db.session.delete(post_to_delete)
    db.session.commit()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
