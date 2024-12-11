## Fine-Tuning ALBERT on SQuAD Dataset
### Project Overview
This project demonstrates the fine-tuning of the ALBERT (A Lite BERT) model for question-answering tasks using the SQuAD dataset.
The objective of this project was to fine-tune the ALBERT model to perform effectively on the SQuAD dataset, a popular benchmark for machine comprehension and question answering. The training process involved optimizing the model to predict answers to questions based on a given context.


### Features
* Utilized Hugging Face's `transformers` library for model fine-tuning.

* Tokenized the SQuAD dataset to prepare it for input into ALBERT.

* Trained and evaluated the model with customizable hyperparameters.

* Leveraged a data collator for dynamic padding during training.


### Model Training
#### Training Arguments
The following training arguments were used for fine-tuning:
```
training_args = TrainingArguments(
    output_dir="my_qa_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=True,
)
```


### Key Constraints
* **Dataset Size**: The SQuAD dataset used was relatively small, which may limit generalization. Incorporating larger datasets or augmenting data could improve performance.
* **System Configuration**: The model was fine-tuned locally on a system with NVIDIA RTX 4050 (6GB). Due to hardware limitations, the batch size was kept small to prevent memory overflow. Users with more powerful configurations are encouraged to increase the batch size or adjust other hyperparameters for better results.


### How to Run
1. Clone the repository:
```
git clone https://github.com/iSathyam31/ALBERT_for_QnA.git
```
2. Install the requirements:
```
pip install -r requirements.txt
```
3. Run the code.


### License
This project is licensed under the MIT License. See the (LICENSE)[LICENSE] file for details.


### Acknowledgments
* Hugging Face for providing the transformers library.

* The creators of the SQuAD dataset for their contributions to advancing natural language understanding

Feel free to explore and modify the project to suit your requirements.