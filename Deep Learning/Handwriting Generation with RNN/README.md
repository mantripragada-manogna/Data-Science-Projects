# Project : Handwriting Synthesis Using LSTM

This project explores the approach for synthesizing handwriting sequences using Long Short-Term Memory Neural Networks. The work is based on a paper from 2014 on Generating Sequences from a Recurrent Neural Networks by Alex Graves. The neural network model will be trained using the “IAM On-Line Handwriting Database” which consists of XML files containing pen point co-ordinates and labeled text. These coordinates and labels are extracted and processed for feeding the model. The Neural Network is implemented with a variant of Recurrent Neural network called LSTM which work well with sequential data. The paper also explores Mixture Density Networks and Soft window network that aid in better training the LSTMs.

## How to run:
1. Clone the repository onto your machine.
2. Download the training data from the following link(http://www.fki.inf.unibe.ch/databases/iam-on-line-handwriting-database/download-the-iam-on-line-handwriting-database). From this page download the file "data/original-xml-part.tar.gz". You will have to signup onto their website before downloading.
3. Extract the downloaded file into a folder named "data" in the cloned repository.
4. Run the following jupyter note book to pre process the data.  (https://github.com/INFO6105-Spring19-02/project-ml-tt18/blob/master/Handwriting_Synthesis/Handwriting%20Analysis.ipynb)
5. Run the "train_model.py" script to train the model on the preprocessed data. It is recommended to train the model using GPU. Preferably use Google Cloud Platform. Follow this documentation to use ML features on GCP (https://github.com/INFO6105-Spring19-02/project-ml-tt18/blob/master/ML%20with%20GCP.docx)
6. Once the model is trained run the folowing jupyter notebook (https://github.com/INFO6105-Spring19-02/project-ml-tt18/blob/master/Handwriting_Synthesis/GenerateHandwriting.ipynb)
7. This jupyter notebook plots the handwritten text for the given text.

## Links:
Research paper: https://github.com/INFO6105-Spring19-02/project-ml-tt18/blob/master/Paper/Handwriting%20Synthesis%20using%20LSTM.pdf </br></br>
Portfolio: https://github.com/INFO6105-Spring19-02/project-ml-tt18/blob/master/Portfolio/Portfolio%20-%20Handwriting%20Synthesis.pdf </br></br>
Project Code: https://github.com/INFO6105-Spring19-02/project-ml-tt18/tree/master/Handwriting_Synthesis </br>
