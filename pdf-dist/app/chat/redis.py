import os
import redis


client = redis.Redis(
    os.environ["REDIS_URI"],
    decode_responses=True,
)