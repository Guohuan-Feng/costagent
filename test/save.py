from azure.storage.blob import BlobClient, ContentSettings
import os

proxy = "http://fun4wx:qawaearata0A%21@rb-proxy-unix-szh.bosch.com:8080"
os.environ["HTTP_PROXY"] = proxy
os.environ["HTTPS_PROXY"] = proxy

container_sas_url = "https://stxchinacwe2edev.blob.core.chinacloudapi.cn/mat?sp=racwd&st=2025-10-29T02:56:01Z&se=2025-11-04T11:11:01Z&skoid=85cc4c22-1860-4056-a7b5-2e7dfe684afc&sktid=6a596574-1518-4214-840e-216bb42592e7&skt=2025-10-29T02:56:01Z&ske=2025-11-04T11:11:01Z&sks=b&skv=2024-11-04&spr=https&sv=2024-11-04&sr=c&sig=uY0MO4eozVkIZPUc8ru%2BSC2C8NaKo3MI3uEIZhdVkoo%3D"

# ✅ 只上传这一份
local_file = "md/CuAl10Ni2_20251027-153804.md"
blob_name = "rates/CuAl10Ni2_20251027-153804.md"


left, _, qs = container_sas_url.partition("?")
blob_sas_url = f"{left.rstrip('/')}/{blob_name}?{qs}"

blob = BlobClient.from_blob_url(blob_sas_url)

print(f"Uploading: {local_file}")
with open(local_file, "rb") as f:
    blob.upload_blob(f, overwrite=True, content_type="text/markdown")
print("Upload OK ->", blob.url.split("?")[0])