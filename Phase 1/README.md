# Principles of Big Data Management 

## Phase 1

#### Goal of the project: 

* To use the tweets dataset and extract all the hastags and URLs in the tweets
* Running the WordCount example in Apache Hadoop and Apache Spark on the extracted hashtags/URLs

#### Directory Tree
```
├── output_py/
├── wordCount/
│   ├── reducer.py
│   ├── mapper.py
├── README.md
├── hashtags_urls.txt
└── logs.txt
```

### Description


* `output_py/`: Contains outputs from Python scripts.
* `wordCount/`: Contains scripts related to word count processing.
* `reducer.py`: Aggregates the occurrences of each word from standard input.
* `mapper.py`: This script extracts hashtags and URLs from JSON-formatted tweets and outputs them as key-value pairs for word count processing.
* `README.md`: Documentation for the project.
* `hashtags_urls.txt`: A file storing hashtags and URLs.
* `logs.txt`: Log file capturing the system or process logs.

--- 
**Project Members: Vani Seth, Rayhan Mahady, and Divya Reddy**
