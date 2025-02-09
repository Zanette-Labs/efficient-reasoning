import time
import ray
import requests
import torch

from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


def request_api_wrapper(url, data, score_key="rewards", try_max_times=5):
    """Synchronous request API wrapper"""
    headers = {
        "Content-Type": "application/json",
    }
    for _ in range(try_max_times):
        try:
            response = requests.post(url=url, json=data, headers=headers, timeout=180)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            response = response.json()
            assert score_key in response, f"{score_key} not in {response}"
            # Return auxiliary information as well not just the score_key
            return response
        except requests.RequestException as e:
            logger.info(f"Request error, please check: {e}")
        except Exception as e:
            logger.info(f"Unexpected error, please check: {e}")
        time.sleep(1)

    raise Exception(f"Request error for {try_max_times} times, returning None. Please check the API server.")


def remote_rm_fn(api_url, queries, steps, total_steps, score_key="rewards"):
    """remote reward model API
    api_url: RM API, We assume that the API supports two modes: merging query + response and not merging
    queries: query+response with the template
    design is made optional.
    score_key: RM score key
    """
    api_outputs = request_api_wrapper(api_url, {"query": queries, "steps": steps, "total_steps": total_steps}, score_key)
    # TODO: Return dict with scores and other information
    return {key: torch.tensor(value) for key, value in api_outputs.items()}


@ray.remote
def remote_rm_fn_ray(api_url, queries, steps, total_steps, score_key="rewards"):
    return remote_rm_fn(api_url, queries, steps, total_steps, score_key)


if __name__ == "__main__":
    # test utils
    url = "http:xxx/get_rm_score"
    score = remote_rm_fn(url, ["example query"], ["example response"])
    print(score)
