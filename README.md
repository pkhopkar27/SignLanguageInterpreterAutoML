# SignLanguageInterpreterAutoML
## Introduction

This solution is implemented as a part of the Altice Innovation Hackathon 2020. 

Challenge Topic - SOCIALLY ENGAGED IN A QUARANTINE WORLD

COVID-19 forced each one of us to be quarantined in our homes. Staying connected and getting day to day things done is a challenge for all of us. The world has moved from real to virtual but a huge impact is made on the lives of people with hearing disabilities. This sudden switch to remote work has highlighted the importance of digital accessibility. Digital accessibility at work, wherever that work takes place, is crucial to the success of every diversity and inclusion initiative. 

“The Sign Language Interpreter that we have implemented in this hackathon is our attempt to improve accessibility for the people with hearing disabilities and make their life easier. 
This solution can be used in various scenarios like while talking to a customer service representative on a video call, an employee talking to another employee in a virtual office meeting or even a judicial proceeding in a virtual courtroom.”  

## Tools and Technologies

- Python
- OpenCV
- Google Cloud Platform
    * AutoML Vision API
    * Buckets
    
## Set up

- Run the following command to install all required dependencies used for this project
  * pyton -m pip r requirements.txt

## How to run the desktop tool?

- Run set_hand_histogram.py to set the hand histogram for creating gestures.
- Once you get a good histogram, save it in the code folder, or you can use the histogram created by us that can be found [here](https://github.com/pkhopkar27/SignLanguageInterpreterAutoML/blob/main/hist).
- Run capture_gestures.py to capture images to be used as training dataset in GCP - AutoML vision API. Images will be saved in gestures folder in the code repository. 
- Now log into Google Cloud Platform annd use GCP- AutoML Vision API to train image classification model. Follow the tutorial [here](https://cloud.google.com/vision/automl/docs/tutorial) to train and deploy your model on GCP.
- Run python_client.py for interpreting the gestures on the live web cam feed. Make sure to change the project id, model id and cloud deployement link in python_client.py before running it.

## Team Members and Contributors

- @bhagya29m 
- @RuchaCB
- @nihir108

## Demo video

[Demo video](https://tinyurl.com/y6x5cu7s)




