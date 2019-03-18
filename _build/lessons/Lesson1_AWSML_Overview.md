---
redirect_from:
  - "/lessons/lesson1-awsml-overview"
interact_link: content/lessons/Lesson1_AWSML_Overview.ipynb
kernel_name: python3
title: 'Lesson 1 - AWS Machine Learning Certification-Overview'
prev_page:
  url: /lessons/lessons
  title: 'Lessons'
next_page:
  url: /lessons/Lesson2_AWSML_Data_Engineering
  title: 'Lesson 2 - Data Engineering for Machine Learning on AWS'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---

<a href="https://colab.research.google.com/github/noahgift/aws-ml-guide/blob/master/Lesson1_AWSML_Overview.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Lesson 1 AWS Machine Learning-Specialty (ML-S) Certification

[Watch Lesson 1:  AWS Machine Learning-Speciality (MLS) Video](https://learning.oreilly.com/videos/aws-certified-machine/9780135556597/9780135556597-ACML_01_01_00)

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


## 1.1 Certification Overview

[Watch 1.1 Certification Overview Video Lesson](https://learning.oreilly.com/videos/aws-certified-machine/9780135556597/9780135556597-ACML_01_01_01)

### Load AWS API Keys

Put keys in local or remote GDrive:  

`cp ~/.aws/credentials /Users/myname/Google\ Drive/awsml/`

#### Mount GDrive




{:.input_area}
```
from google.colab import drive
drive.mount('/content/gdrive')
```


{:.output .output_stream}
```
Drive already mounted at /content/gdrive; to attempt to forcibly remount, call drive.mount("/content/gdrive", force_remount=True).

```



{:.input_area}
```
import os;os.listdir("/content/gdrive/My Drive/awsml")
```





{:.output .output_data_text}
```
['kaggle.json', 'credentials', 'config']
```



#### Install Boto



{:.input_area}
```
!pip -q install boto3

```


#### Create API Config



{:.input_area}
```
!mkdir -p ~/.aws &&\
  cp /content/gdrive/My\ Drive/awsml/credentials ~/.aws/credentials 
```


#### Test Comprehend API Call



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
   'date': 'Tue, 11 Dec 2018 01:07:46 GMT',
   'x-amzn-requestid': '2b726af9-fce1-11e8-a4e1-2d227516436e'},
  'HTTPStatusCode': 200,
  'RequestId': '2b726af9-fce1-11e8-a4e1-2d227516436e',
  'RetryAttempts': 0},
 'Sentiment': 'NEUTRAL',
 'SentimentScore': {'Mixed': 0.008628507144749165,
  'Negative': 0.1037612184882164,
  'Neutral': 0.8582549691200256,
  'Positive': 0.0293553676456213}}
```



### Audience



Intended for individuals who perform Development or Data Science role

### Validates ability to:

* "design, implement, deploy and maintain ML solutions"
* "select and justify appropriate ML approaches"
* "indentify appropriate AWS services to implement ML solutions"
* "Design and implement scalable, cost-optimized, reliable, and secure ML solutions"

### Recommended AWS Knowledge

* Hands-on experience:

 - developing
 - architecting
 - running ML/deep learning workloads on AWS Cloud
 
* Ability to express intuition behind basic ML algorithms
* Experience performance basic hyperparameter optimization
* Experience with ML and deep learning frameworks
* Ability to follow model training best practices
* Ability to follow deployment and operational best practices

## 1.2 Exam Study Resources

[Watch 1.2 Exam Study Resources Video Lesson](https://learning.oreilly.com/videos/aws-certified-machine/9780135556597/9780135556597-ACML_01_01_02)

### Official AWS Material

![Sequential Split](https://user-images.githubusercontent.com/58792/49745472-de7cb500-fc53-11e8-8dd8-1fb2075d1373.png)

* [AWS ML University](https://aws.amazon.com/training/learning-paths/machine-learning/)
* [Machine Learning on AWS](https://aws.amazon.com/machine-learning/)
* [Amazon Machine Learning Concepts](https://docs.aws.amazon.com/machine-learning/latest/dg/amazon-machine-learning-key-concepts.html)
* [Splitting Your Data](https://docs.aws.amazon.com/machine-learning/latest/dg/splitting-types.html)
* [Types of Machine Learning Models](https://docs.aws.amazon.com/machine-learning/latest/dg/types-of-ml-models.html)
* [Data Transformations](https://docs.aws.amazon.com/machine-learning/latest/dg/data-transformations-reference.html)
* [Containerized Machine Learning on AWS Video](https://www.youtube.com/watch?v=Jw9CpQkCvpM)
* [AWS re:Invent Machine Learning Talk 2017](https://www.youtube.com/watch?v=Q7N2iVfgA0U&list=PLOS8MNIsAkpJCaOSQP6t5uoVtwUr0SFvM)

## 1.3 Review Exam Guide

[Watch 1.3 Review Exam Guide Video Lesson](https://learning.oreilly.com/videos/aws-certified-machine/9780135556597/9780135556597-ACML_01_01_03)

[AWS Certified Machine Learning–Specialty (MLS-C01) Exam Guide ](https://d1.awsstatic.com/training-and-certification/machinelearning/AWS%20Certified%20Machine%20Learning%20-%20Specialty_Exam%20Guide_v1.1_FINAL.pdf)

### Domains Covered

#### Data Engineering

* Create data repositories for Machine Learning
* Identify and implement a data-ingestion solution
* Identify and implement a data-transformation solution


#### Exploratory Data Analysis

* Sanitize and prepare data modeling
* Perform feature engineering
* Analyze and visualize data for machine learning

#### Modeling



*   Frame business problems as machine learning problems
*   Select the appropriate model(s) for a given machine learning problem
* Train machine learning models
* Perform hyperparameter optimization
* Evaluate machine learning models



#### Machine Learning Implementation and Operations

*   Build machine learning solutions for performance, availability, scalability, resiliency, and fault tolerance.
*  Recommend and implement the appropriate machine learning services and features for a given problem.
* Apply basic AWS security practices to machine learning solutions.
* Deploy and operationalize machine learning solutions.



## 1.4 Exam Strategy

[Watch 1.4 Exam Strategy Video Lesson](https://learning.oreilly.com/videos/aws-certified-machine/9780135556597/9780135556597-ACML_01_01_04)

### Official Exam Study Guide

![Study Path](https://user-images.githubusercontent.com/58792/49443332-06b27280-f781-11e8-8ec2-88af4d47724e.png)

## 1.5 Best Practices of ML on AWS

[Watch 1.5 Best Practices of ML on AWS Video Lesson](https://learning.oreilly.com/videos/aws-certified-machine/9780135556597/9780135556597-ACML_01_01_05)

### Build vs Buy

![hero](https://user-images.githubusercontent.com/58792/45260894-08e67a00-b3a8-11e8-941f-e539cb10f8e1.jpg)

### Overview

Auto and Managed ML Services on AWS 

*   AWS Sagemaker
*   AWS Machine Learning

Human in the Loop

* AWS Sagemaker Ground Truth
* AWS Mechanical Turk

Supporting Technologies

* MXNet



### AWS Sagemaker

[Sagemaker Features](https://aws.amazon.com/sagemaker/features/)

![AWS Sagemaker](https://user-images.githubusercontent.com/58792/45426785-09a23900-b652-11e8-9ba6-4ff58a8103d5.png)


*   Build

 - Preconfigured Jupyter Notebooks
 - Built in, High Performance Algorithms (Optimized petabyte-scale peformance)
 - Supports many frameworks:  MXnet, Tensorflow, sklearn


*   Train

 - One click train with S3 target
 - Automatic hyperparameter tuning
 - Can scale training to clusters of machines with single line of code

*   Deploy

  - One click Deploy (creates HTTPS endpoint)
  - Automatic A/B Testing
  - Fully managed auto-scaling of inference
  



### AWS Machine Learning

[Amazon Machine Learning Service](https://aws.amazon.com/aml/)

![alt text](https://user-images.githubusercontent.com/58792/45425471-53892000-b64e-11e8-8895-794014ce3702.png)

**Key Features**


*   " makes it easy for developers of all skill levels to use machine learning technology"
*   Managaged ML Service





**Resources**
*   [Introduction to AWS Machine Learning Video](https://youtu.be/PAHU8tPA7xs)
*   [Kaggle + Amazon ML Service](https://aws.amazon.com/blogs/big-data/building-a-numeric-regression-model-with-amazon-machine-learning/)
* [amazon ml dev guide](https://docs.aws.amazon.com/machine-learning/latest/dg/what-is-amazon-machine-learning.html)


### MXNet

[Open Source Docs](https://gluon.mxnet.io/)

![mxnet logo](https://user-images.githubusercontent.com/58792/45424774-5e42b580-b64c-11e8-90f3-1f022fbc35e7.png)


*  Alternative to TensorFlow
*  Amazon Backed
*  Integrated into Sagemaker and DeepLense
*  Has production-ready orchestration tools:  ECS, Docker, Kubernetes 
*  Can do[ inference via AWS Lambda](https://github.com/awslabs/mxnet-lambda) 

**Resources**

* [Serving Machine Learning Models with Apache MXNet & AWS Fargate](https://www.youtube.com/watch?v=YlUoa3hLz78)
* [mxnet-lambda](https://github.com/awslabs/mxnet-lambda)





### Demo AWS Sagemaker






### Demo AWS ML Service






#### Build vs Buy

* When to use AI APIs vs ML Platforms

## 1.6 Techniques to accelerate hands-on practice

[Watch 1.6 Techniques to accelerate hands-on practice Video Lesson](https://learning.oreilly.com/videos/aws-certified-machine/9780135556597/9780135556597-ACML_01_01_06)

### Using Interactive QWIKLABS


A temporary, but real AWS environment that teaches specific labs.

![QWiklabs](https://user-images.githubusercontent.com/58792/48742665-b2ea4a00-ec14-11e8-8805-0600f60e637b.png)


#### Big Data on AWS

[Big Data on AWS Quest](https://run.qwiklabs.com/quests/5)



#### Intro to Machine Learning
[Introduction to Machine Learning](https://run.qwiklabs.com/focuses/275?parent=catalog)


### Using AWS Console, Sagemaker and APIS




*   Console to Explore
*   Sagemaker
*   APIs via Boto



## 1.7 Understand important ML related services

[Watch 1.7 Understand important ML related services Video Lesson](https://learning.oreilly.com/videos/aws-certified-machine/9780135556597/9780135556597-ACML_01_01_07)

### ML Platform Services

#### Amazon Sagemaker

**[Demo] Overview Console**

#### AWS DeepLens

![alt text](https://user-images.githubusercontent.com/58792/45308029-688e7380-b4d5-11e8-8ffb-9422184c274c.png)

**[Demo] Overview Console**

#### EMR

* [Amazon EMR](https://aws.amazon.com/emr/?nc2=type_a)
* Managed Hadoop/Spark
* Use cases:

  - clickstream analysis
  - real-time analytics
  - log analysis
  - ETL
  - predictive analytics
  - genomics



**[Demo] Overview Console**

#### Amazon Machine Learning Service

* [Amazon Machine Learning](https://aws.amazon.com/aml/?nc2=type_a)
* Provides visualization and wizards to help create ML models
* AutoML technology

**[Demo] Overview Console**

### Deep Learning on AWS

#### AWS Deep Learning AMIs

* [AWS Deep Learning AMIs](https://aws.amazon.com/machine-learning/amis/)

#### Apache MXNet



*   [Apache MXNet on AWS](https://aws.amazon.com/machine-learning/amis/)




#### TensorFlow on AWS

* [TensorFlow on AWS](https://aws.amazon.com/tensorflow/)
* [88% of TensorFlow workloads are on AWS](https://d1.awsstatic.com/whitepapers/nucleus-tensorflow.pdf)

### AI APIs

#### Vision Services

##### Rekognition Image

* Deep learning-based image analysis

##### Rekognition Video

* Deep learning-based video analysis

#### Conversational chatbots

##### Amazon Lex

* [Amazon Lex](https://aws.amazon.com/lex/)
* Service for building conversational interfaces

#### Language Services

##### Comprehend

* [AWS Comprehend](https://aws.amazon.com/comprehend/)
* NLP service
  - Sentiment Analysis


**Neutral Sentiment**



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
   'date': 'Tue, 11 Dec 2018 01:07:59 GMT',
   'x-amzn-requestid': '3368fdfa-fce1-11e8-8656-5735cedac40a'},
  'HTTPStatusCode': 200,
  'RequestId': '3368fdfa-fce1-11e8-8656-5735cedac40a',
  'RetryAttempts': 0},
 'Sentiment': 'NEUTRAL',
 'SentimentScore': {'Mixed': 0.008628507144749165,
  'Negative': 0.1037612184882164,
  'Neutral': 0.8582549691200256,
  'Positive': 0.0293553676456213}}
```



**Negative Sentiment**



{:.input_area}
```
import boto3
comprehend = boto3.client(service_name='comprehend', region_name="us-east-1")
text = "There is smoke in San Francisco and it makes me very angry"
comprehend.detect_sentiment(Text=text, LanguageCode='en')
```





{:.output .output_data_text}
```
{'ResponseMetadata': {'HTTPHeaders': {'connection': 'keep-alive',
   'content-length': '164',
   'content-type': 'application/x-amz-json-1.1',
   'date': 'Tue, 11 Dec 2018 01:08:17 GMT',
   'x-amzn-requestid': '3e127c6f-fce1-11e8-b936-d1be599a3739'},
  'HTTPStatusCode': 200,
  'RequestId': '3e127c6f-fce1-11e8-b936-d1be599a3739',
  'RetryAttempts': 0},
 'Sentiment': 'NEGATIVE',
 'SentimentScore': {'Mixed': 0.00937745813280344,
  'Negative': 0.9539545774459839,
  'Neutral': 0.03619137033820152,
  'Positive': 0.00047663392615504563}}
```





{:.input_area}
```
trump_text = """
which it was not (but even if it was, it is only a CIVIL CASE, like Obama’s - but it was done correctly by a lawyer and there would not even be a fine. Lawyer’s liability if he made a mistake, not me). Cohen just trying to get his sentence reduced. WITCH HUNT!

"""
comprehend = boto3.client(service_name='comprehend', region_name="us-east-1")
comprehend.detect_sentiment(Text=trump_text,LanguageCode='en')

```





{:.output .output_data_text}
```
{'ResponseMetadata': {'HTTPHeaders': {'connection': 'keep-alive',
   'content-length': '160',
   'content-type': 'application/x-amz-json-1.1',
   'date': 'Tue, 11 Dec 2018 01:10:42 GMT',
   'x-amzn-requestid': '94b5be74-fce1-11e8-92e1-212d40a34fc0'},
  'HTTPStatusCode': 200,
  'RequestId': '94b5be74-fce1-11e8-92e1-212d40a34fc0',
  'RetryAttempts': 0},
 'Sentiment': 'NEGATIVE',
 'SentimentScore': {'Mixed': 0.0772324725985527,
  'Negative': 0.776195228099823,
  'Neutral': 0.10541720688343048,
  'Positive': 0.04115508496761322}}
```



**Positive Sentiment**



{:.input_area}
```
import boto3
comprehend = boto3.client(service_name='comprehend', region_name="us-east-1")
text = "There is no more smoke in San Francisco and it makes me very happy"
comprehend.detect_sentiment(Text=text, LanguageCode='en')
```





{:.output .output_data_text}
```
{'ResponseMetadata': {'HTTPHeaders': {'connection': 'keep-alive',
   'content-length': '163',
   'content-type': 'application/x-amz-json-1.1',
   'date': 'Mon, 10 Dec 2018 22:41:39 GMT',
   'x-amzn-requestid': 'c25e7c7c-fccc-11e8-b966-5f330c378b33'},
  'HTTPStatusCode': 200,
  'RequestId': 'c25e7c7c-fccc-11e8-b966-5f330c378b33',
  'RetryAttempts': 0},
 'Sentiment': 'POSITIVE',
 'SentimentScore': {'Mixed': 0.01339163538068533,
  'Negative': 0.007987627759575844,
  'Neutral': 0.04016350954771042,
  'Positive': 0.9384573101997375}}
```



##### Translate

* [Amazon Translate](https://aws.amazon.com/translate/?nc2=type_a)
* Natural and accurate language translation



{:.input_area}
```
import boto3
client = boto3.client('translate', region_name="us-east-1")
text = "Hola, mi nombre es Noah y me encanta el aprendizaje automático."
client.translate_text(Text=text,SourceLanguageCode="auto", TargetLanguageCode="en")

```





{:.output .output_data_text}
```
{'ResponseMetadata': {'HTTPHeaders': {'connection': 'keep-alive',
   'content-length': '124',
   'content-type': 'application/x-amz-json-1.1',
   'date': 'Tue, 11 Dec 2018 01:11:05 GMT',
   'x-amzn-requestid': 'a1dd350d-fce1-11e8-88f8-15e276ec8e51'},
  'HTTPStatusCode': 200,
  'RequestId': 'a1dd350d-fce1-11e8-88f8-15e276ec8e51',
  'RetryAttempts': 0},
 'SourceLanguageCode': 'es',
 'TargetLanguageCode': 'en',
 'TranslatedText': 'Hello, my name is Noah and I love machine learning.'}
```



##### Transcribe

* [Amazon Transcribe](https://aws.amazon.com/transcribe/)
* Automatic speech recognition
* [Translate Tutorial](https://aws.amazon.com/getting-started/tutorials/create-audio-transcript-transcribe/)




{:.input_area}
```
import boto3
s3_path = "https://s3.amazonaws.com/pai-transcribe/transcribe-sample.dac1d22492611d998262c8c856b98a74180a1a8f.mp3"
```


##### Polly


* [Amazon Polly](https://aws.amazon.com/polly/)
* Turns text into lifelike speech
