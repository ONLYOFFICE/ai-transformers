# requires following additional package: requests
import requests

# URL to the raw files from llama.cpp GitHub repo
repo_url = 'https://raw.githubusercontent.com/ggerganov/llama.cpp/master'
# script names
filenames = ['convert_hf_to_gguf.py', 'convert_hf_to_gguf_update.py']

for filename in filenames:
    url = repo_url + '/' + filename
    try:
        print(f'Downloading script "{filename}"...')
        # send HTTP GET request to fetch the files
        response = requests.get(url)
        # check for HTTP errors
        response.raise_for_status()

        # write the content to a local file
        with open(filename, 'wb') as file:
            file.write(response.content)

        print(f'Script "{filename}" downloaded successfully.')

    except requests.exceptions.RequestException as e:
        print(f'Failed to download the script "{filename}": {e}')
