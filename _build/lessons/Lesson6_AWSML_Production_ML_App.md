---
redirect_from:
  - "/lessons/lesson6-awsml-production-ml-app"
interact_link: content/lessons/Lesson6_AWSML_Production_ML_App.ipynb
kernel_name: python3
title: 'Lesson 6 - Amazon Machine Learning Production Machine Learning Applications'
prev_page:
  url: /lessons/Lesson5_AWSML_Operationalization
  title: 'Lesson 5 - Amazon Machine Learning Operationalization'
next_page:
  url: /lessons/Lesson7_AWSML_Case_Studies
  title: 'Lesson 7 - Amazon Machine Learning Case Studies'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---

<a href="https://colab.research.google.com/github/noahgift/aws-ml-guide/blob/master/Lesson6_AWSML_Production_ML_App.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Lesson 6:  Create a Production Machine Learning Application

[Watch Lesson 6:  Create a Production Machine Learning Application Video](https://learning.oreilly.com/videos/aws-certified-machine/9780135556597/9780135556597-ACML_01_06_00)

## Pragmatic AI Labs



![alt text](https://paiml.com/images/logo_with_slogan_white_background.png)

This notebook was produced by [Pragmatic AI Labs](https://paiml.com/).  You can continue learning about these topics by:

*   Buying a copy of [Pragmatic AI: An Introduction to Cloud-Based Machine Learning](http://www.informit.com/store/pragmatic-ai-an-introduction-to-cloud-based-machine-9780134863863) from Informit.
*   Buying a copy of  [Pragmatic AI: An Introduction to Cloud-Based Machine Learning](https://www.amazon.com/Pragmatic-AI-Introduction-Cloud-Based-Learning/dp/0134863860) from Amazon
*   Reading an online copy of [Pragmatic AI:Pragmatic AI: An Introduction to Cloud-Based Machine Learning](https://www.safaribooksonline.com/library/view/pragmatic-ai-an/9780134863924/)
*  Watching video [Essential Machine Learning and AI with Python and Jupyter Notebook-Video-SafariOnline](https://www.safaribooksonline.com/videos/essential-machine-learning/9780135261118) on Safari Books Online.
* Watching video [AWS Certified Machine Learning-Speciality](https://learning.oreilly.com/videos/aws-certified-machine/9780135556597)
* Purchasing video [Essential Machine Learning and AI with Python and Jupyter Notebook- Purchase Video](http://www.informit.com/store/essential-machine-learning-and-ai-with-python-and-jupyter-9780135261095)
*   Viewing more content at [noahgift.com](https://noahgift.com/)


## Load AWS API Keys

Put keys in local or remote GDrive:  

`cp ~/.aws/credentials /Users/myname/Google\ Drive/awsml/`

### Mount GDrive




{:.input_area}
```
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
```


{:.output .output_stream}
```
Mounted at /content/gdrive

```



{:.input_area}
```
import os;os.listdir("/content/gdrive/My Drive/awsml")
```





{:.output .output_data_text}
```
['kaggle.json', 'credentials', 'config']
```



### Install Boto



{:.input_area}
```
!pip -q install boto3

```


### Create API Config



{:.input_area}
```
!mkdir -p ~/.aws &&\
  cp /content/gdrive/My\ Drive/awsml/credentials ~/.aws/credentials 
```


### Test Comprehend API Call



{:.input_area}
```
import boto3
comprehend = boto3.client(service_name='comprehend', region_name="us-east-1")
text = "There is smoke in San Francisco"
comprehend.detect_sentiment(Text=text, LanguageCode='en')
```





{:.output .output_data_text}
```
{'ResponseMetadata': {'HTTPHeaders': {'connection': 'keep-alive',
   'content-length': '160',
   'content-type': 'application/x-amz-json-1.1',
   'date': 'Thu, 22 Nov 2018 00:21:54 GMT',
   'x-amzn-requestid': '9d69a0a9-edec-11e8-8560-532dc7aa62ea'},
  'HTTPStatusCode': 200,
  'RequestId': '9d69a0a9-edec-11e8-8560-532dc7aa62ea',
  'RetryAttempts': 0},
 'Sentiment': 'NEUTRAL',
 'SentimentScore': {'Mixed': 0.008628507144749165,
  'Negative': 0.1037612184882164,
  'Neutral': 0.8582549691200256,
  'Positive': 0.0293553676456213}}
```



## 6.1 Create Machine Learning Data Pipeline

### Data Pipeline Patterns

![data_engineers](https://user-images.githubusercontent.com/58792/49955060-c062bf00-feb6-11e8-8204-79d629ab2862.png)

#### [Anti-Pattern] Accidental Data Pipeline


* Production SQL Server is center

#### [Pattern] Batch Data Lake

* Batch ML Pipeline to and from Data Lake

#### [Pattern]  Batch Infinite Resources on Production

* Environment uses 

## 6.2 EDA with Sagemaker

### [Demo] County Census Clustering

* [Original AWS Article on Census Clustering](https://aws.amazon.com/blogs/machine-learning/analyze-us-census-data-for-population-segmentation-using-amazon-sagemaker/)

## 6.3 Create Machine Model using AWS Sagemaker

### [Demo] Sagemaker Pipeline

## 6.4 Deploy Machine Learning Model using AWS Sagemaker

### [Demo] Sagemaker Pipeline
