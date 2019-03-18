---
redirect_from:
  - "/lessons/lesson5-awsml-operationalization"
interact_link: content/lessons/Lesson5_AWSML_Operationalization.ipynb
kernel_name: python3
title: 'Lesson 5 - Amazon Machine Learning Operationalization'
prev_page:
  url: /lessons/Lesson4_AWSML_Modeling
  title: 'Lesson 4 - Amazon Machine Learning Modeling'
next_page:
  url: /lessons/Lesson6_AWSML_Production_ML_App
  title: 'Lesson 6 - Amazon Machine Learning Production Machine Learning Applications'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---

<a href="https://colab.research.google.com/github/noahgift/aws-ml-guide/blob/master/Lesson5_AWSML_Operationalization.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Lesson 5:  Operationalize Machine Learning on AWS

[Watch Lesson 5:  Operationalize Machine Learning on AWS Video](https://learning.oreilly.com/videos/aws-certified-machine/9780135556597/9780135556597-ACML_01_05_00)

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



## 5.1 Understand ML Operations

### Key Concepts



*   Monitoring
*   Security
*   Retraining Models
*   A/B Testing
*   TCO (Total Cost of Ownership)




### MLOPS


* Are you using a simple enough model?
* Are you using the Data Lake or wired directly into production SQL DB?
* Do you have alerts setup for prediction threshold failures?
* Environments?  Dev, Stage, Prod

## 5.2 Use Containerization with Machine Learning and Deep Learning

### Key Concepts

![docker_workflows](https://user-images.githubusercontent.com/58792/49868900-9b415400-fdc3-11e8-807f-375bbe0a4684.png)

#### Amazon ECS (Elastic Container Service)

##### [Demo] ECS



*   Create a repo
*   List item



#### Amazon EKS (Kubernetes on AWS)

## 5.3 Implement continuous deployment and delivery for Machine Learning

### Key Concepts

![codebuild](https://user-images.githubusercontent.com/58792/49869955-da24d900-fdc6-11e8-925f-767fb7fb522f.png)


### [Demo] Code Build

* buildspec.yml
* console
* build job
* sync to s3
* ECS integration

## 5.4 A/B Testing production deployments

### Key Concepts


* Sagemaker A/B Testing Capabilities
* Deciding on ratio of delivery to ML Model

### [Demo] Sagemaker A/B

## 5.5 Troubleshoot Production Deployment

### Key Concepts

* Using Cloudwatch
* Searching Cloudwatch Logs
* Alerting on key events
* Using Auto-Scale Capabilities
* Enterprise AWS Support

#### [Demo]Cloudwatch Features

## 5.6 Production Security

#### Key Concepts

* Understanding KMS system (Encryption)
* IAM Roles for Sagemaker
* IAM Roles for VPC


### [Demo] Sagemaker Security Features

## 5.7 Cost and Efficiency of ML Systems

#### Key Concepts

* Understanding Spot Instances (show spot code)
* Understanding Proper use of CPU vs GPU Resources
* Scale up and Scale Down
* Improve Time to Market
* Choosing wisely from AI API vs "Do it Yourself"


### [Demo] Spot Instances on AWS


