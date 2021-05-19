docker build -t alexeyshevchuk/models:v1 .

docker run -p 8000:8000 alexeyshevchuk/models:v1

python make_request.py 
