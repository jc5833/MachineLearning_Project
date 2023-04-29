# MachineLearning_Project
Topic Research: Research past work related to your project
Describe your project objective
The objective of our project is to make a model that is able to analyze images and determine whether the image is of a car or something else
What are potential applications of your project?
The applications of our project is that it could be used for image detection, the main use of that would be for stuff like captcha. 
What are the known challenges of building or implementing a model for your topic?
Image training models require a big dataset and we do not have computers made for processing huge amounts of data. Github stores a max of 1000 documents in 1 folder so our database will be limited. Any kind of variation in the image can cause the model to be less accurate since it  might be looking for specific features that also leads to the model overfitting.
What types of datasets have been used in the past?
In the past the types of datasets used for image processing tasks have been face datasets to distinguish people, a MIT dataset to learn what is indoors and what is not, scenery classification done by intel, and image datasets for medical purposes.
What types of methods have been applied to related research in the past?
There have been tests done to see if image filtering had any effect on the efficiency of the model, giving the model a specific trait and having it learn off that trait, and image segmentation to divide the image into portions in order to learn more accurately.
What is the state of the art (SOTA) method for your ML task?
The SOTA method for image processing depends on the application and what needs to be analyzed, however Convolutional neural networks have been shown to be very effective in image processing. The other method that is used a lot is FixEfficientNet which combines previously used models FixRes and EfficientNet.
What metrics are used for measuring model success in this task?
The metrics used for model success are the accuracy, by using the ratio of success/total we can tell how successful the model was. Then there is also the Intersection over Union which measures the area of overlap/area of union.
Dataset: Address the following about your dataset
Describe the dataset for your project.
The dataset we used was from a kaggle dataset that was for this kind of task, the dataset had about 10,000 images that were vehicles and 10,000 images that were not vehicles and those would be used to train a model accurately. However for us we did not have the benefit of being able to run all 20,000 images in 1 model so we limited it down to around 1,700 for our version, however in Github we were only able to upload 1,000 images for each (vehicle and non vehicle).
What are some challenges of the dataset you are working with?
The amount of images we were given and the storage space that comes along with so many images was a challenge. Additionally, resolution was not the best so the model would have some issues determining key features unlike if the images were in 4k quality.
How was your dataset prepared (i.e. how was data collected)? How was it labeled (if at all)? (If you cannot find this information, just mention so)
I could not find how the dataset was collected but they way it was labeled was already divided into vehicles and non-vehicles which made it easier for us when inputting the data to the model.
Are there any potential biases embedded in your dataset that may lead to problems if ever used in production?
The key issue we could see is that since there is not too big of a dataset, compared to the task, it might end up underfitting and not being able to accurately predict the information.
Data Analysis: Perform an analysis of your dataset. (Remember this must not be done on the test set).
Provide statistics about your dataset to give a rough overview of your data in numbers
The statistics that we used to have an overview of our data is counting how many images there were in each data set. 
Explain what insight each statistic provides about your dataset with regards to the ML task
This statistics helps us determine what type of layers we are going to use, and how many in order to effectively teach the model.
Data Cleaning: Address the following questions about data cleaning…
Implement at least 2 data cleaning techniques to your dataset
We ensured there were no duplicates in the data set, and we made sure the data was somewhat balanced.
Explain each data cleaning method and why you chose to apply it to your dataset
The first one we didn't want to have duplicates to avoid false learning of the model, and we wanted the dataset to be balanced to give better results. 
Data Processing: Transform your dataset in ways that would be useful for your project objective.
Implement at least 3 data transformation methods to your dataset
Rescale, horizonta_flip, zoom_range.
Explain each data processing method and why you chose to apply it to your dataset
The purpose is to give the model more variations to learn from to have a better accuracy. 
Model Implementation: Implement ML models for your task
Implement at least 3 significantly different ML models for your task (You don’t need to implement models from scratch. Using existing python libraries is allowed)
For our image classification program we used a convolutional neural network using the TensorFlow and Keras libraries. 
Explain how each model works and why you chose to use it for your project (Explain each model in detail with accompanying visuals if needed)
This model is very popular in image classification, because it uses both image sets to distinguish between vehicles and non-vehicles and then tries to predict whether an image is a vehicle or not in the testing phase. 
Explain the strengths and weaknesses of each model you selected
Some of the strengths of this model are that it can learn from raw images without needing manual feature engineering, can handle high dimensional images because of good spatial correlations within images. Some weaknesses include, they can require large amounts of memory to run, and can easily overfit if the model is too complex. 
Model Training and Tuning: Train and tune your models as you train them
Did any of your models ever overfit? What did you do to address this?
Yes, when we started we began with each data set being 500 images. We then decided to add more images to counter the overfitting. 
Did any of your models underfit? What did you do to address this?
No.
For each model, which hyperparameters did you tune and what effect did those changes have on the model performance?
We always kept the same ones, however when we saw that the validation accuracy was the same we decided to see what change deleting some hyperparameters did and it would perform even worse. Instead we tweaked our data sets.  


Results: Training your models, perform a final test and estimate your model performance (Give the results with tables or graphs rather than answering the questions below in words)
How do you measure the accuracy model? Why is this metric(s) good?

refer to image#1 and image#2

What is the training time of your best models?
Epoch 1/3 102/102 [==============================] - 437s 4s/step - loss: 0.2622 - accuracy: 0.8972 - val_loss: 0.0942 - val_accuracy: 0.9691 
Epoch 2/3 102/102 [==============================] - 418s 4s/step - loss: 0.1102 - accuracy: 0.9635 - val_loss: 0.0869 - val_accuracy: 0.9715 
Epoch 3/3 102/102 [==============================] - 416s 4s/step - loss: 0.0516 - accuracy: 0.9819 - val_loss: 0.0338 - val_accuracy: 0.9902
What is the size (memory) of your best model?

What model performed the best in each of the above criteria?
The CNN.
Include images of sample outputs of your model
We've included these in question 8.
Discussion: After training, tuning, and testing your models, do a post analysis of your experiments and models you have created
Was there a single model that clearly performed the best?
Yes, CNN,
Why do you think the models that performed the best were the most successful?
I believe that because of a combination of multiple things, having sufficient data, good architecture, and making sure the correct hyperparameters were used with the right amount of tuning was crucial to this. 
What data transformations or hyperparameter settings lead to the most successful improvements in model performance?
We believe Conv2D and MaxPooling were the most beneficial to the models positive performance. 
Were the above changes that lead to improvements also observed in any of research related to your project
Yes, in some of the references that we used it was explained how these previous hyperparameters led to efficient model performance within CNN models. 
If you were to continue this project, describe more modeling, training, or advanced validation techniques
Continue tweaking the parameters and look into using Conv2d and Maxpooling further in order to have a more precise model.
Note any other interesting observations or findings from your experiments
None.
What potential problems or additional work in improving or testing your model do you foresee if you planned to deploy it to production?
We would need to have the model trained on the full 20,000 dataset in order to have a more precise model.
If you were to deploy your model in production, how would you make your models' decisions more interpretable?
Make it so that once the model is trained we could input an image and have the model determine whether that is a vehicle or not.
References: Include references to research, tutorials, or other resources you used throughout your project

https://www.kaggle.com/datasets/brsdincer/vehicle-detection-image-set
https://datagen.tech/guides/image-datasets/image-datasets/
https://www.folio3.ai/blog/ml-image-classification-datasets/
https://www.v7labs.com/blog/image-processing-guide
https://towardsdatascience.com/state-of-the-art-image-classification-algorithm-fixefficientnet-l2-98b93deeb04c
https://www.analyticsvidhya.com/blog/2021/06/evaluate-your-model-metrics-for-image-classification-and-detection/

