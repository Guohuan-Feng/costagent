import requests

url = "http://localhost:8000/upload_corpus"   # ← 换成你的接口地址

data = {
    'project': 'mat',
    'options': '{"source":"process_rate"}'
}

files = {
    'files': open(r"C:\Users\WUT5WX\Downloads\process_rate.csv", 'rb')  # 文件路径自己改
}

response = requests.post(url, data=data, files=files)
print(response.status_code)
print(response.text)
