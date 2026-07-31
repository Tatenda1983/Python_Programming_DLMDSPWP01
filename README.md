# Programming With Python Assignment-DLMDSPWP01

## My Python Application

A Python Application For Ideal Function Selection And Mapping With Docker Containerization.

## Requirements

- Docker Desktop

## Setup

Build the Docker image:

​```bash
docker build -t myapp:latest .
​```

## Run

​```bash
docker run --rm myapp:latest
​```

## Run Tests

​```bash
docker run --rm myapp:latest pytest
​```

## Dependencies

- numpy
- pandas
- bokeh
- pytest