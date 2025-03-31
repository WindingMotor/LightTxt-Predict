import requests
import json
import time

def example_client():
    """Example client that uses the LLM prediction server"""
    base_url = "http://localhost:8000"
    
    # Wait for server to be ready
    print("Waiting for the LLM prediction server to be ready...")
    server_ready = False
    while not server_ready:
        try:
            response = requests.get(f"{base_url}/health", timeout=1)
            data = response.json()
            
            if data["model_ready"]:
                server_ready = True
                print("Server is ready!")
            else:
                print("Model still loading, waiting...")
                time.sleep(2)
        except requests.exceptions.RequestException:
            print("Server not responding yet, waiting...")
            time.sleep(2)
    
    # Example 1: Simple prediction with GET
    text = "I would like to"
    print(f"\n1. Getting predictions for: '{text}'")
    
    response = requests.get(f"{base_url}/predict?text={text}")
    data = response.json()
    
    print("Predictions:")
    for pred in data["predictions"]:
        print(f"  • {pred['word']} ({pred['probability']:.0%})")
    print(f"Model time: {data['metadata']['model_time_ms']:.1f}ms")
    
    # Example 2: Prediction with POST request
    text = "The quick brown fox"
    print(f"\n2. Getting predictions with POST for: '{text}'")
    
    response = requests.post(
        f"{base_url}/predict",
        json={"text": text, "top_k": 5}
    )
    data = response.json()
    
    print("Predictions:")
    for pred in data["predictions"]:
        print(f"  • {pred['word']} ({pred['probability']:.0%})")
    
    # Example 3: Batch predictions
    texts = ["Hello world", "Once upon a", "The weather is"]
    print(f"\n3. Getting batch predictions for multiple inputs")
    
    try:
        response = requests.post(
            f"{base_url}/predict",
            json={"text": texts, "top_k": 3}
        )
        data = response.json()
        
        for i, (text, result) in enumerate(zip(texts, data)):
            print(f"\nPredictions for '{text}':")
            for pred in result["predictions"]:
                print(f"  • {pred['word']} ({pred['probability']:.0%})")
                
    except Exception as e:
        print(f"Error with batch predictions: {e}")
    
    # Example 4: Get server stats
    print("\n4. Getting server statistics")
    
    response = requests.get(f"{base_url}/stats")
    stats = response.json()
    
    print(f"Total requests: {stats['requests']}")
    print(f"Average prediction time: {stats['avg_prediction_time']*1000:.1f}ms")
    print(f"Server uptime: {stats['uptime_seconds']:.0f} seconds")

if __name__ == "__main__":
    example_client()