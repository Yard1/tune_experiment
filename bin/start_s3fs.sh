mkdir results
s3fs tune-experiment-result results -o iam_role="auto" -o url="https://s3-us-west-2.amazonaws.com" &