#!/bin/bash
# 1. Force the terminal to change directory directly into your project folder
cd /Users/aadhirachhibber/Desktop/API_Assignment

echo "=========================================" >> pipeline_execution.log
echo "Execution Timestamp: $(date)" >> pipeline_execution.log
echo "=========================================" >> pipeline_execution.log

# 2. Run python using the explicit local script path
/opt/anaconda3/bin/python /Users/aadhirachhibber/Desktop/API_Assignment/pipeline_utils.py >> pipeline_execution.log 2>&1