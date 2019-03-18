---
redirect_from:
  - "/lessons/lesson2-awsml-data-engineering"
interact_link: content/lessons/Lesson2_AWSML_Data_Engineering.ipynb
kernel_name: python3
title: 'Lesson 2 - Data Engineering for Machine Learning on AWS'
prev_page:
  url: /lessons/Lesson1_AWSML_Overview
  title: 'Lesson 1 - AWS Machine Learning Certification-Overview'
next_page:
  url: /lessons/Lesson3_AWSML_Exploratory_Data_Analysis
  title: 'Lesson 3 - Amazon Machine Learning Exploratory Data Analysis'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---

<a href="https://colab.research.google.com/github/noahgift/aws-ml-guide/blob/master/Lesson2_AWSML_Data_Engineering.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Lesson 2 Data Engineering for ML on AWS

[Watch Lesson 2:  Data Engineering for ML on AWS Video](https://learning.oreilly.com/videos/aws-certified-machine/9780135556597/9780135556597-ACML_01_02_00)

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




{:.input_area}
```
import os;os.listdir("/content/gdrive/My Drive/awsml")
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


## 2.1 Data Ingestion Concepts

### Data Lakes

**Central Repository** for all data at any scale

![data_lake](https://user-images.githubusercontent.com/58792/49777724-8aef8300-fcb6-11e8-981e-96d14498a801.png)

#### AWS Lake Formation

* New Service Announced at Reinvent 2018
* Build a secure lake in days...**not months**
* Enforce security policies
* Gain and manage insights

![aws_lake](https://user-images.githubusercontent.com/58792/49777834-f9ccdc00-fcb6-11e8-84a0-7295a0c69a15.png)

### Kinesis (STREAMING)

**Solves Three Key Problems**



*   Time-series Analytics
*   Real-time Dashboards
*   Real-time Metrics



#### Kinesis Analytics Workflow
![Kinesis Analytics](https://user-images.githubusercontent.com/58792/49440264-02ce2280-f778-11e8-9d7e-149819e74807.png)

#### Kinesis Real-Time Log Analytics Example

![Real-Time Log Analytics](https://user-images.githubusercontent.com/58792/49440433-7cfea700-f778-11e8-8cd5-55999cb7713c.png)

#### Kinesis Ad Tech Pipeline

![Ad Tech Pipeline](https://user-images.githubusercontent.com/58792/49441021-285c2b80-f77a-11e8-82e2-da9006dc4c6d.png)

#### Kinesis IoT

![Kinesis IoT](https://user-images.githubusercontent.com/58792/49441101-5e011480-f77a-11e8-9727-4f7706361a08.png)

#### [Demo] Kinesis

### AWS Batch (BATCH)

Example could be Financial Service Trade Analysis

![financial_services_trade](https://user-images.githubusercontent.com/58792/49778503-64334b80-fcba-11e8-85e7-dcdbfe473cd9.png)

#### Using AWS Batch for ML Jobs

* *[Watch Video Lesson 11.6:  Use AWS Batch for ML Jobs](https://www.safaribooksonline.com/videos/essential-machine-learning/9780135261118/9780135261118-EMLA_01_11_06)*


https://aws.amazon.com/batch/

![alt text](https://d1.awsstatic.com/Test%20Images/Kate%20Test%20Images/Dilithium-Diagrams_Visual-Effects-Rendering.ad9c0479c3772c67953e96ef8ae76a5095373d81.png)


Example submissions tool

```python
@cli.group()
def run():
    """Run AWS Batch"""

@run.command("submit")
@click.option("--queue", default="first-run-job-queue", help="Batch Queue")
@click.option("--jobname", default="1", help="Name of Job")
@click.option("--jobdef", default="test", help="Job Definition")
@click.option("--cmd", default=["uname"], help="Container Override Commands")
def submit(queue, jobname, jobdef, cmd):
    """Submit a job"""

    result = submit_job(
        job_name=jobname,
        job_queue=queue,
        job_definition=jobdef,
        command=cmd
    )
    click.echo("CLI:  Run Job Called")
    return result
```

### Lambda (EVENTS)


* Serverless
*   Used in most if not all ML Platforms
 - DeepLense
 - Sagemaker
 - S3 Events



#### Starting development with AWS Python Lambda development with Chalice

* *[Watch Video Lesson 11.3:  Use AWS Lambda development with Chalice](https://www.safaribooksonline.com/videos/essential-machine-learning/9780135261118/9780135261118-EMLA_01_11_03)*



***Demo on Sagemaker Terminal***

https://github.com/aws/chalice

*Hello World Example:*

```python
$ pip install chalice
$ chalice new-project helloworld && cd helloworld
$ cat app.py

from chalice import Chalice

app = Chalice(app_name="helloworld")

@app.route("/")
def index():
    return {"hello": "world"}

$ chalice deploy
...
https://endpoint/dev

$ curl https://endpoint/api
{"hello": "world"}
```

References:

[Serverless Web Scraping Project](https://github.com/noahgift/web_scraping_python)

#### [Demo] Deploying Hello World Lambda Function

### Using Step functions with AWS

* *[Watch Video Lesson 11.5:  Use AWS Step Functions](https://www.safaribooksonline.com/videos/essential-machine-learning/9780135261118/9780135261118-EMLA_01_11_05)*

https://aws.amazon.com/step-functions/

![Step Functions](https://d1.awsstatic.com/product-marketing/Step%20Functions/AmazonCloudWatchUpdated4.a57e968b08739e170aa504feed8db3761de21e60.png)

Example Project:

https://github.com/noahgift/web_scraping_python

[Demo] Step Function

## 2.2 Data Cleaning and Preparation

### Ensuring High Quality Data



*   Validity
*   Accuracy
*   Completeness
*   Consistency
*   Uniformity



### Dealing with missing values

Often easy way is to drop missing values




{:.input_area}
```
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/noahgift/real_estate_ml/master/data/Zip_Zhvi_SingleFamilyResidence.csv")
df.isnull().sum()
```





{:.output .output_data_text}
```
RegionID         1
RegionName       1
City             1
State            1
Metro         1140
CountyName       1
SizeRank         1
1996-04       4440
1996-05       4309
1996-06       4285
1996-07       4278
1996-08       4265
1996-09       4265
1996-10       4265
1996-11       4258
1996-12       4258
1997-01       4212
1997-02       3588
1997-03       3546
1997-04       3546
1997-05       3545
1997-06       3543
1997-07       3543
1997-08       3357
1997-09       3355
1997-10       3353
1997-11       3347
1997-12       3341
1998-01       3317
1998-02       3073
              ... 
2015-04         13
2015-05          1
2015-06          1
2015-07          1
2015-08          1
2015-09          2
2015-10          3
2015-11          1
2015-12          1
2016-01          1
2016-02         19
2016-03         19
2016-04         19
2016-05         19
2016-06          1
2016-07          1
2016-08          1
2016-09          1
2016-10          1
2016-11          1
2016-12         51
2017-01          1
2017-02          1
2017-03          1
2017-04          1
2017-05          1
2017-06          1
2017-07          1
2017-08          1
2017-09          1
Length: 265, dtype: int64
```





{:.input_area}
```
df2 = df.dropna()
df2.isnull().sum()
```





{:.output .output_data_text}
```
RegionID      0
RegionName    0
City          0
State         0
Metro         0
CountyName    0
SizeRank      0
1996-04       0
1996-05       0
1996-06       0
1996-07       0
1996-08       0
1996-09       0
1996-10       0
1996-11       0
1996-12       0
1997-01       0
1997-02       0
1997-03       0
1997-04       0
1997-05       0
1997-06       0
1997-07       0
1997-08       0
1997-09       0
1997-10       0
1997-11       0
1997-12       0
1998-01       0
1998-02       0
             ..
2015-04       0
2015-05       0
2015-06       0
2015-07       0
2015-08       0
2015-09       0
2015-10       0
2015-11       0
2015-12       0
2016-01       0
2016-02       0
2016-03       0
2016-04       0
2016-05       0
2016-06       0
2016-07       0
2016-08       0
2016-09       0
2016-10       0
2016-11       0
2016-12       0
2017-01       0
2017-02       0
2017-03       0
2017-04       0
2017-05       0
2017-06       0
2017-07       0
2017-08       0
2017-09       0
Length: 265, dtype: int64
```



### Cleaning Wikipedia Handle Example



```python
"""
Example Route To Construct:
https://wikimedia.org/api/rest_v1/ +
metrics/pageviews/per-article/ +
en.wikipedia/all-access/user/ +
LeBron_James/daily/2015070100/2017070500 +
"""
import requests
import pandas as pd
import time
import wikipedia

BASE_URL =\
 "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user"

def construct_url(handle, period, start, end):
    """Constructs a URL based on arguments
    Should construct the following URL:
    /LeBron_James/daily/2015070100/2017070500 
    """

    
    urls  = [BASE_URL, handle, period, start, end]
    constructed = str.join('/', urls)
    return constructed

def query_wikipedia_pageviews(url):

    res = requests.get(url)
    return res.json()

def wikipedia_pageviews(handle, period, start, end):
    """Returns JSON"""

    constructed_url = construct_url(handle, period, start,end)
    pageviews = query_wikipedia_pageviews(url=constructed_url)
    return pageviews

def wikipedia_2016(handle,sleep=0):
    """Retrieve pageviews for 2016""" 
    
    print("SLEEP: {sleep}".format(sleep=sleep))
    time.sleep(sleep)
    pageviews = wikipedia_pageviews(handle=handle, 
            period="daily", start="2016010100", end="2016123100")
    if not 'items' in pageviews:
        print("NO PAGEVIEWS: {handle}".format(handle=handle))
        return None
    return pageviews

def create_wikipedia_df(handles):
    """Creates a Dataframe of Pageviews"""

    pageviews = []
    timestamps = []    
    names = []
    wikipedia_handles = []
    for name, handle in handles.items():
        pageviews_record = wikipedia_2016(handle)
        if pageviews_record is None:
            continue
        for record in pageviews_record['items']:
            pageviews.append(record['views'])
            timestamps.append(record['timestamp'])
            names.append(name)
            wikipedia_handles.append(handle)
    data = {
        "names": names,
        "wikipedia_handles": wikipedia_handles,
        "pageviews": pageviews,
        "timestamps": timestamps 
    }
    df = pd.DataFrame(data)
    return df    


def create_wikipedia_handle(raw_handle):
    """Takes a raw handle and converts it to a wikipedia handle"""

    wikipedia_handle = raw_handle.replace(" ", "_")
    return wikipedia_handle

def create_wikipedia_nba_handle(name):
    """Appends basketball to link"""

    url = " ".join([name, "(basketball)"])
    return url

def wikipedia_current_nba_roster():
    """Gets all links on wikipedia current roster page"""

    links = {}
    nba = wikipedia.page("List_of_current_NBA_team_rosters")
    for link in nba.links:
        links[link] = create_wikipedia_handle(link)
    return links

def guess_wikipedia_nba_handle(data="data/nba_2017_br.csv"):
    """Attempt to get the correct wikipedia handle"""

    links = wikipedia_current_nba_roster() 
    nba = pd.read_csv(data)
    count = 0
    verified = {}
    guesses = {}
    for player in nba["Player"].values:
        if player in links:
            print("Player: {player}, Link: {link} ".format(player=player,
                 link=links[player]))
            print(count)
            count += 1
            verified[player] = links[player] #add wikipedia link
        else:
            print("NO MATCH: {player}".format(player=player))
            guesses[player] = create_wikipedia_handle(player)
    return verified, guesses

def validate_wikipedia_guesses(guesses):
    """Validate guessed wikipedia accounts"""

    verified = {}
    wrong = {}
    for name, link in guesses.items():
        try:
            page = wikipedia.page(link)
        except (wikipedia.DisambiguationError, wikipedia.PageError) as error:
            #try basketball suffix
            nba_handle = create_wikipedia_nba_handle(name)
            try:
                page = wikipedia.page(nba_handle)
                print("Initial wikipedia URL Failed: {error}".format(error=error))
            except (wikipedia.DisambiguationError, wikipedia.PageError) as error:
                print("Second Match Failure: {error}".format(error=error))
                wrong[name] = link
                continue
        if "NBA" in page.summary:
            verified[name] = link
        else:
            print("NO GUESS MATCH: {name}".format(name=name))
            wrong[name] = link
    return verified, wrong

def clean_wikipedia_handles(data="data/nba_2017_br.csv"):
    """Clean Handles"""

    verified, guesses = guess_wikipedia_nba_handle(data=data)
    verified_cleaned, wrong = validate_wikipedia_guesses(guesses)
    print("WRONG Matches: {wrong}".format(wrong=wrong))
    handles = {**verified, **verified_cleaned}
    return handles

def nba_wikipedia_dataframe(data="data/nba_2017_br.csv"):
    handles = clean_wikipedia_handles(data=data)
    df = create_wikipedia_df(handles)    
    return df

def create_wikipedia_csv(data="data/nba_2017_br.csv"):
    df = nba_wikipedia_dataframe(data=data)
    df.to_csv("data/wikipedia_nba.csv")


if __name__ == "__main__":
    create_wikipedia_csv() 
```



### Related AWS Services

These services could all help prepare and clean data


*   AWS Glue
*   AWS Machine Learning
*   AWS Kinesis
*   AWS Lambda
*   AWS Sagemaker



## 2.3 Data Storage Concepts

### Database Overview



![Database Styles](https://user-images.githubusercontent.com/58792/48925585-2214a800-ee7a-11e8-8546-767177679328.png)

* [One size database doesn't fit anyone](https://www.allthingsdistributed.com/2018/06/purpose-built-databases-in-aws.html)

### Using AWS DynamoDB

* *[Watch Video Lesson 11.4:  Use AWS DynamoDB](https://www.safaribooksonline.com/videos/essential-machine-learning/9780135261118/9780135261118-EMLA_01_11_04)*

https://aws.amazon.com/dynamodb/

![alt text](https://d1.awsstatic.com/video-thumbs/dynamodb/AWS-online-games-wide.ada4247744e9be9a6d857b2e13b7eb78b18bf3a5.png)

Query Example:

```python
def query_police_department_record_by_guid(guid):
    """Gets one record in the PD table by guid
    
    In [5]: rec = query_police_department_record_by_guid(
        "7e607b82-9e18-49dc-a9d7-e9628a9147ad"
        )
    
    In [7]: rec
    Out[7]: 
    {'PoliceDepartmentName': 'Hollister',
     'UpdateTime': 'Fri Mar  2 12:43:43 2018',
     'guid': '7e607b82-9e18-49dc-a9d7-e9628a9147ad'}
    """
    
    db = dynamodb_resource()
    extra_msg = {"region_name": REGION, "aws_service": "dynamodb", 
        "police_department_table":POLICE_DEPARTMENTS_TABLE,
        "guid":guid}
    log.info(f"Get PD record by GUID", extra=extra_msg)
    pd_table = db.Table(POLICE_DEPARTMENTS_TABLE)
    response = pd_table.get_item(
        Key={
            'guid': guid
            }
    )
    return response['Item']
```


#### [Demo] DynamoDB

### Redshift

* Data Warehouse Solution for AWS
* Column Data Store (Great at counting large data)

## 2.4 Learn ETL Solutions (Extract-Transform-Load)

### AWS Glue

#### AWS Glue is fully managed ETL Service

![AWS Glue Screen](https://user-images.githubusercontent.com/58792/49441953-dff23d00-f77c-11e8-9065-dab53c47c345.png)

#### AWS Glue Workflow



*   Build Data Catalog
*   Generate and Edit Transformations
*   Schedule and Run Jobs



#### [DEMO] AWS Glue

### EMR

* Can be used for large scale distributed data jobs

### Athena

* Can replace many ETL
* Serverless
* Built on Presto w/ SQL Support
* Meant to query Data Lake

#### [DEMO] Athena

### Data Pipeline

*  create complex data processing workloads that are fault tolerant, repeatable, and highly available

#### [Demo] Data Pipeline

## 2.5 Batch vs Streaming Data

### Impact on ML Pipeline

* More control of model training in batch (can decide when to retrain)
* Continuously retraining model could provide better prediction results or worse results
 - Did input stream suddenly get more users or less users?
 - Is there an A/B testing scenario?

### Batch

*   Data is batched at intervals
*   Simplest approach to create predictions
*   Many Services on AWS Capable of Batch Processing
 - AWS Glue
 - AWS Data Pipeline
 - AWS Batch
 - EMR





### Streaming


* Continously polled or pushed
* More complex method of prediction
* Many Services on AWS Capable of Streaming
 - Kinesis
 - IoT

## 2.6 Data Security

### AWS KMS (Key Management Service)



*   Integrated with AWS Encryption SDK
*   CloudTrail gives independent view of who accessed encrypted data



### AWS Cloud Trail

![cloud_trail](https://user-images.githubusercontent.com/58792/49812752-f834ff80-fd1a-11e8-9ad6-bafa8e1b0779.png)



*  enables governance, compliance, operational auditing
*  visibility into user and resource activity
*  security analysis and troubleshooting
*  security analysis and troubleshooting



#### [Demo] Cloud Trail

### Other Aspects

* IAM Roles
* Security Groups
* VPC

## 2.7 Data Backup and Recovery

### Most AWS Services Have Snapshot and Backup Capabilities

* RDS
* S3
* DynamoDB

### S3 Backup and Recovery


* S3 Snapshots
* Amazon Glacier archive

### [Demo] S3 Snapshot Demo

