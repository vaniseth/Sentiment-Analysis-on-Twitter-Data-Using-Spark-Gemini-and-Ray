# Hadoop Streaming Word Count Program: Troubleshooting and Testing Guide

## Overview
This guide provides step-by-step instructions for setting up, testing, and running a Python-based word count program using Hadoop Streaming. It includes commands, troubleshooting tips, and expected outputs to ensure smooth execution.

---

## 1. Environment and Prerequisites
### Platform:
- Mac (M3 chip)
- Hadoop Version: 3.4.1

### Required Scripts:
- **mapper.py**
- **reducer.py**

Ensure both scripts are correctly set up before running the job.

---

## 2. Preparation Steps
### 2.1 File Permissions and Shebang

Make sure the Python scripts are executable:
```bash
chmod +x mapper.py reducer.py
```

Check that the first line of each script is:
```python
#!/usr/bin/env python3
```

---

## 3. Testing Locally

### 3.1 Test the Mapper Script
To verify that `mapper.py` is functioning correctly:

```bash
echo "hello world" | python3 mapper.py
```
**Expected output:**
```
hello   1
world   1
```

For a larger file:
```bash
cat sample_input.txt | python3 mapper.py
```
**Expected output (word frequency pairs):**
```
hi,    1
i       1
am      1
rayhan. 1
```

### 3.2 Test the Reducer Script

Simulate Hadoop's sorting step before feeding input to the reducer:
```bash
cat sample_input.txt | python3 mapper.py | sort | python3 reducer.py
```

**Expected aggregated output:**
```
Bangladesh.  1
am           2
from         1
hi,          1
i            2
rayhan.      1
```

---

## 4. HDFS File Management

### 4.1 Upload Input File to HDFS
Create an input directory if it does not exist:
```bash
hadoop fs -mkdir -p /user/bigData/input
```

Upload the input file:
```bash
hadoop fs -put hashtags_urls.txt /user/bigData/input/
```

Verify the upload:
```bash
hadoop fs -ls /user/bigData/input
```

---

## 5. Running the Hadoop Streaming Job

### 5.1 Remove Existing Output Directory
Hadoop does not overwrite an existing output directory. If the output directory exists, remove it:
```bash
hadoop fs -rm -r /user/bigData/output
```

### 5.2 Run the Hadoop Streaming Job
Execute the following command:
```bash
hadoop jar /Users/mhr6wb/hadoop-3.4.1/share/hadoop/tools/lib/hadoop-streaming-3.4.1.jar \
    -files mapper.py,reducer.py \
    -mapper "./mapper.py" \
    -reducer "./reducer.py" \
    -input /user/bigData/input/hashtags_urls.txt \
    -output /user/bigData/output
```

### Explanation of Command:
- `-files mapper.py,reducer.py` → Distributes the scripts to Hadoop nodes.
- `-mapper "./mapper.py"` → Runs the mapper script.
- `-reducer "./reducer.py"` → Runs the reducer script.
- `-input` → Specifies the input file location in HDFS.
- `-output` → Specifies the output directory in HDFS.

---

### 5.3 Monitor and Verify Job Execution

#### Track Job Progress:
Hadoop provides a web-based interface to track the job:
- **Resource Manager:** [http://localhost:8088/proxy/application_...](http://localhost:8088/proxy/application_...)
- **HDFS Explorer:** [http://localhost:9870/explorer.html#/user/bigData/output](http://localhost:9870/explorer.html#/user/bigData/output)

#### View Output After Completion:
```bash
hadoop fs -cat /user/bigData/output/part-*
```

#### Retrieve Output to Local System:
```bash
hdfs dfs -get /user/bigData/output ./
```

---

## 6. Expected Logs
If everything runs successfully, you should see logs similar to the following:
```log
2025-03-07 15:07:20,877 WARN util.NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
Deleted /user/bigData/output_py
...
2025-03-07 15:07:26,178 INFO impl.YarnClientImpl: Submitted application application_1741341400678_0013
2025-03-07 15:07:26,192 INFO mapreduce.Job: The url to track the job: http://localhost:8088/proxy/application_1741341400678_0013/
2025-03-07 15:07:44,478 INFO mapreduce.Job: Job job_1741341400678_0013 completed successfully
...
2025-03-07 15:07:44,564 INFO streaming.StreamJob: Output directory: /user/bigData/output_py
```

---

## 7. Troubleshooting

| Issue | Possible Cause | Solution |
|--------|----------------|----------|
| `Permission denied` error when running scripts | Scripts lack execute permission | Run `chmod +x mapper.py reducer.py` |
| No output in reducer | Mapper output not sorted | Ensure to use `sort` before piping to reducer |
| `Output directory already exists` error | Output directory is not removed | Run `hadoop fs -rm -r /user/bigData/output` before executing the job |
| `Unable to load native-hadoop library` warning | Mac-specific compatibility issue | This is a warning, and it does not affect job execution |

---

## 8. Conclusion
This guide provides a structured approach to setting up, testing, and running a Hadoop Streaming word count job. Follow each step carefully to troubleshoot issues and ensure a smooth execution.

