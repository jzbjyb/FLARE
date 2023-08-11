from typing import List
import json
import os
import requests
import logging
from ratelimit import limits, sleep_and_retry
from flask import Flask, request, jsonify
from diskcache import Cache
logging.basicConfig(level=logging.INFO)


subscription_key = os.environ['BING_SEARCH_KEY']
cache = Cache('.bing_search_cache')
app = Flask(__name__)


@sleep_and_retry
@limits(calls=50, period=1)
def bing_search(
    query: str,
    only_domain: str = None,
    exclude_domains: List[str] = [],
    endpoint: str = 'https://api.bing.microsoft.com/v7.0/search',
):
    # modify query based on domain
    if only_domain is not None:
        query = query + f' site:{only_domain}'
    elif exclude_domains:
        query = query + ' ' + ' '.join([f'-site:{d}' for d in exclude_domains])

    # cache hit
    if query in cache:
        logging.info(f'CACHE: {query}')
        return cache[query]
    else:
        # construct the request
        params = {
            'q': query,
            'mkt': 'en-US',
            'responseFilter': ['Webpages'],
            'count': 50,
            'safeSearch': 'Off',
            'setLang': 'en-US'
        }

        # construct the header
        headers = {
            'Ocp-Apim-Subscription-Key': subscription_key ,
            'Pragma': 'no-cache',
            'X-MSEdge-ClientID':'05EFBDF2399564422335AF4538A8655D',
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; WOW64; Tridentz/7.0; Touch; rv:11.0) like Gecko',
            'X-MSEdge-ClientIP': "128.2.211.82"
        }

        # call the API
        try:
            response = requests.get(endpoint, headers=headers, params=params)
            response.raise_for_status()
            result = response.json()
            cache[query] = result
            return(result)
        except Exception as ex:
            logging.error(f'EXCEPTION: {ex}')
            return None


@app.route('/bing_search', methods=['POST'])
def bing_request():
    headers = request.headers
    data = request.get_json()
    query = data.get('query', None)
    only_domain = data.get('+domain', None)
    exclude_domains = data.get('-domain', '')
    exclude_domains = exclude_domains.split(',') if exclude_domains else []
    result = bing_search(
        query=query,
        only_domain=only_domain,
        exclude_domains=exclude_domains)
    if result is None:
        return jsonify({'message': 'ERROR: Request Failed'}), 404
    else:
        return jsonify(result), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
