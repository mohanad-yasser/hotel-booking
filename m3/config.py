def read_config(path="config.txt"):
    config = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            config[key.strip()] = value.strip()
    return config


cfg = read_config()

NEO4J_URI = cfg.get("URI")
NEO4J_USERNAME = cfg.get("USERNAME")
NEO4J_PASSWORD = cfg.get("PASSWORD")

HF_API_TOKEN = cfg.get("HF_API_TOKEN")
HF_MODEL_ID = cfg.get("HF_MODEL_ID")

if not NEO4J_URI or not NEO4J_USERNAME or not NEO4J_PASSWORD:
    raise ValueError("Missing Neo4j config values in config.txt")