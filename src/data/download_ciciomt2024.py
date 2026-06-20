import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm
import concurrent.futures

# URL of the dataset index
BASE_URL = "https://cicresearch.ca/IOTDataset/CICIoMT2024/browse.php?p=WiFI_and_MQTT"

# Directory where files will be saved
DOWNLOAD_DIR = "ciciomt2024_dataset"

# Headers to bypass basic 403 Forbidden checks and authenticate with CIC session
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Cookie": "Token=k7gk1qjttu7k5cqm0ou8lat2j8"
}

def download_file(file_url, dest_path):
    """Downloads a single file with a progress bar."""
    if os.path.exists(dest_path):
        print(f"[SKIP] {os.path.basename(dest_path)} already exists.")
        return

    try:
        response = requests.get(file_url, headers=HEADERS, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024 # 1 MB
        
        with open(dest_path, 'wb') as f, tqdm(
            desc=os.path.basename(dest_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                f.write(data)
                bar.update(len(data))
    except Exception as e:
        print(f"[ERROR] Failed to download {file_url}: {e}")

def main():
    print(f"Fetching index page: {BASE_URL}")
    response = requests.get(BASE_URL, headers=HEADERS)
    
    if response.status_code != 200:
        print(f"Error fetching page: {response.status_code}")
        return
        
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all links that look like downloadable files
    # Typically they end in .csv, .pcap, .zip, etc., or point to a download.php endpoint
    download_links = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        # You can adjust this filter based on what the actual links look like.
        # Often they contain 'download.php' or end with '.csv' / '.pcap'
        if href.endswith('.csv') or href.endswith('.pcap') or href.endswith('.zip') or 'download' in href.lower():
            full_url = urljoin(BASE_URL, href)
            # Remove duplicates
            if full_url not in download_links:
                download_links.append(full_url)
                
    if not download_links:
        print("No download links found! The site structure might require manual inspection.")
        return
        
    print(f"Found {len(download_links)} files to download.")
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    # Download in parallel using threads
    max_workers = 4 # Adjust this depending on your network and the server's rate limits
    print(f"Starting parallel download with {max_workers} workers...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for url in download_links:
            # Try to guess a filename from the URL
            filename = url.split('/')[-1]
            if "=" in filename:
                filename = filename.split('=')[-1]
                
            dest_path = os.path.join(DOWNLOAD_DIR, filename)
            futures.append(executor.submit(download_file, url, dest_path))
            
        # Wait for all to complete
        concurrent.futures.wait(futures)

    print("All downloads finished!")

if __name__ == "__main__":
    main()
