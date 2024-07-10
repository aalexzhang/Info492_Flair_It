# Flair-It

Flair-It is a machine learning project aimed to automate and streamline the flairing system for Reddit. With the disruption of manual flairing, many subreddits face challenges in categorizing posts effectively. Flair-It provides an intelligent platform that helps automate the assignment of flairs to Reddit posts, enhancing user experience and improving content discoverability. Check out Flair-It through our demo.

## Link to Research/Report Paper
https://docs.google.com/document/d/1u8Ry1eBghT_vFedIspSBHiGwB2JFvkZd8Lz2HI_NIqs/edit?usp=sharing

## Features

### Automated Flairing
- **Advanced NLP Models**: Flair-It utilizes RoBERTa transformer models to automatically assign flairs to Reddit posts based on their content.
- **Efficient Categorization**: Enhances the organization and discoverability of posts within subreddits.

### Data Pipeline
- **Data Collection and Processing**: Robust ETL processes for gathering and processing subreddit data.
- **Model Training**: Efficient training pipeline to fine-tune RoBERTa models on subreddit-specific data.

### Demo Website
- **Interactive Interface**: The website showcases the results of the trained models, providing a user-friendly interface to explore how the flairing system works.
- **Real-Time Flairing**: Demonstrates the automatic assignment of flairs in a media forum format.

## Usage

Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/yourusername/Flair-It.git
cd Flair-It
pip install -r requirements.txt
```
### Data Collection
Download Reddit post data from [ArcticShift](https://github.com/ArthurHeitmann/arctic_shift).
### Data Cleaning
Use [Json_to_CSV.ipynb](Week5/Json_to_CSV.ipynb) and [DatasetsCreation.ipynb](Week6/DatasetsCreation.ipynb) to clean the post data.

### Model Training
Run [finalized_roberta.ipynb](./finalized_roberta.ipynb) using your cleaned post data, you may need to use a cloud notebook to run the model training. We recommend Kaggle. Push the model to HuggingFace using an access token, using these [instructions](https://huggingface.co/docs/hub/en/models-uploading).

### Additions to Demo
Add a new subreddit under [config.json](website/config.json) along with the flairs used to train the model under "label_mapping".

Run the demo:

```bash
cd website
python app.py
```
Visit `localhost:5000`. Your subreddit should show up in the left sidebar, and submitting a post with a title and a body will create a new post with the model's suggested flair.

Our demo currently works with r/udub, r/usc, r/uiuc, r/rutgers, and r/nyu.

## Contributing
If you would like to add more to our project, here are some steps to do so.

1. Fork the repository.
2. Create a new branch for your feature.
3. Commit your changes and push the branch to your fork.
4. Open a pull request with a detailed description of your changes.
