import requests

url = "https://cicresearch.ca/IOTDataset/CICIoMT2024/browse.php?p=WiFI_and_MQTT"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive"
}

try:
    r = requests.get(url, headers=headers)
    with open("cic_html.txt", "w", encoding="utf-8") as f:
        f.write(f"Status: {r.status_code}\n")
        f.write(r.text)
    print(f"Status: {r.status_code}. Saved to cic_html.txt")
except Exception as e:
    print(e)
