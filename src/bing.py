from typing import List, Dict
import json
import requests

api_url = "http://127.0.0.1:8989/bing_search"

def search_bing_api(
        query: str,
        only_domain: str = None,
        exclude_domains: List[str] = [],
        count: int = 10,
    ):
    # prepare request
    data = {'query': query}
    if only_domain:
        data['+domain'] = only_domain
    elif exclude_domains:
        data['-domain'] = ','.join(exclude_domains)
    data = json.dumps(data)
    headers = {'X-Api-Key': 'jzbjybnb', 'Content-Type': 'application/json'}
    try:
        # sent request
        response = requests.request('POST', api_url, headers=headers, data=data)
        response = response.json()
        results: List[Dict] = []
        # collect results
        if 'webPages' in response and 'value' in response['webPages']:
            for page in response['webPages']['value']:
                result = {
                    'url': page['url'],
                    'title': page['name'],
                    'snippet': page['snippet']
                }
                exclude = False
                if exclude_domains:
                    for d in exclude_domains:
                        if d in page['url']:
                            exclude = True
                            break
                if not exclude:
                    results.append(result)
        return results
    except Exception as ex:
        raise ex


def search_bing_batch(queries: List[str], **kwargs):
    results: List[List[Dict]] = []
    for query in queries:
        results.append(search_bing_api(query, **kwargs))
    return results
