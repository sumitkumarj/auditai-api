# Lambda Python 3.11 base image
FROM public.ecr.aws/lambda/python:3.11

# Set working directory within the image
WORKDIR /var/task

# Copy dependency manifest and install
COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt --target .

# Copy application code
COPY app ./app
COPY template.yaml ./template.yaml

# Set the CMD to your handler
# (Lambda entry is provided by base image; CMD here is for local docker runs)
CMD ["app.main.lambda_handler"]
