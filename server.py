import sys
import time
import threading
import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

# Import the Predictor from the existing file
from predictor import Predictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("llm-server")

class LLMPredictionServer:
    """Ultra-lightweight HTTP server for LLM-based predictive text"""
    
    def __init__(self, host="localhost", port=8000, model_optimizations=True):
        """Initialize the prediction server"""
        self.host = host
        self.port = port
        self.predictor = None
        self.predictor_ready = False
        self.stats = {
            "requests": 0,
            "total_prediction_time": 0,
            "avg_prediction_time": 0,
            "start_time": time.time()
        }
        
        # Model optimization settings
        self.model_optimizations = model_optimizations
        
        # Initialize predictor in a separate thread to keep server responsive
        logger.info("Starting LLM Prediction Server")
        logger.info(f"Loading model in background...")
        self.load_predictor_thread = threading.Thread(target=self._load_predictor)
        self.load_predictor_thread.daemon = True
        self.load_predictor_thread.start()
    
    def _load_predictor(self):
        """Load the LLM predictor in a separate thread"""
        try:
            start_time = time.time()
            optimizations = {
                "quantize": self.model_optimizations,
                "precompute_common": self.model_optimizations,
                "use_half_precision": self.model_optimizations
            }
            
            self.predictor = Predictor(**optimizations)
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            self.predictor_ready = True
        except Exception as e:
            logger.error(f"Error loading predictor: {e}")
            sys.exit(1)
    
    def start(self):
        """Start the HTTP server"""
        # Create custom request handler with access to the predictor
        server_instance = self  # Reference to the server instance
        
        class PredictorHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                # Override to use our logger instead of printing to stderr
                if args[0] == "GET" and ("/health" in args[1] or "/stats" in args[1]):
                    return  # Don't log health checks to reduce noise
                logger.info(f"{self.address_string()} - {format % args}")
            
            def _send_response(self, status_code, data=None, content_type="application/json"):
                self.send_response(status_code)
                self.send_header("Content-Type", content_type)
                self.send_header("Access-Control-Allow-Origin", "*")  # CORS support
                self.end_headers()
                
                if data:
                    if content_type == "application/json":
                        self.wfile.write(json.dumps(data).encode("utf-8"))
                    else:
                        self.wfile.write(data.encode("utf-8"))
            
            def do_OPTIONS(self):
                # Handle CORS preflight requests
                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.end_headers()
            
            def do_GET(self):
                # Parse URL and query parameters
                parsed_url = urlparse(self.path)
                path = parsed_url.path
                query_params = parse_qs(parsed_url.query)
                
                # Health check endpoint
                if path == "/health":
                    self._send_response(200, {
                        "status": "up", 
                        "model_ready": server_instance.predictor_ready
                    })
                    return
                
                # Stats endpoint
                elif path == "/stats":
                    self._send_response(200, server_instance.stats)
                    return
                
                # Predict endpoint with GET (for simple requests)
                elif path == "/predict":
                    if not server_instance.predictor_ready:
                        self._send_response(503, {"error": "Model is still loading"})
                        return
                    
                    # Extract parameters
                    if "text" not in query_params:
                        self._send_response(400, {"error": "Missing 'text' parameter"})
                        return
                    
                    text = query_params["text"][0]
                    top_k = int(query_params.get("top_k", [3])[0])
                    
                    # Get predictions
                    try:
                        predictions = server_instance._get_predictions(text, top_k)
                        self._send_response(200, predictions)
                    except Exception as e:
                        logger.error(f"Error generating predictions: {e}")
                        self._send_response(500, {"error": str(e)})
                
                # Handle unknown endpoints
                else:
                    self._send_response(404, {"error": "Endpoint not found"})
            
            def do_POST(self):
                # Currently only support /predict endpoint for POST
                if self.path != "/predict":
                    self._send_response(404, {"error": "Endpoint not found"})
                    return
                
                if not server_instance.predictor_ready:
                    self._send_response(503, {"error": "Model is still loading"})
                    return
                
                # Read request body
                content_length = int(self.headers.get("Content-Length", 0))
                if content_length == 0:
                    self._send_response(400, {"error": "Empty request body"})
                    return
                
                request_body = self.rfile.read(content_length).decode("utf-8")
                
                try:
                    # Parse JSON body
                    body = json.loads(request_body)
                    
                    # Extract parameters
                    if "text" not in body:
                        self._send_response(400, {"error": "Missing 'text' parameter"})
                        return
                    
                    text = body["text"]
                    top_k = body.get("top_k", 3)
                    
                    # Get batch predictions if multiple texts provided
                    if isinstance(text, list):
                        batch_results = server_instance._get_batch_predictions(text, top_k)
                        self._send_response(200, batch_results)
                    else:
                        # Get single prediction
                        predictions = server_instance._get_predictions(text, top_k)
                        self._send_response(200, predictions)
                        
                except json.JSONDecodeError:
                    self._send_response(400, {"error": "Invalid JSON format"})
                except Exception as e:
                    logger.error(f"Error processing request: {e}")
                    self._send_response(500, {"error": str(e)})
        
        # Start the server
        server = HTTPServer((self.host, self.port), PredictorHandler)
        logger.info(f"Server started at http://{self.host}:{self.port}")
        logger.info("Available endpoints:")
        logger.info(f"  GET  /health - Server health check")
        logger.info(f"  GET  /stats - Server statistics")
        logger.info(f"  GET  /predict?text=<text>&top_k=<k> - Get predictions")
        logger.info(f"  POST /predict - Get predictions with JSON body")
        
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Server stopped by user")
        finally:
            server.server_close()
    
    def _get_predictions(self, text, top_k=3):
        """Get predictions using the loaded model"""
        server_start = time.time()
        
        # Get predictions from the model
        suggestions, model_time = self.predictor.predict_next(text, top_k=top_k)
        
        # Format the result
        results = {
            "predictions": [{"word": word, "probability": prob} for word, prob in suggestions],
            "metadata": {
                "model_time_ms": model_time * 1000,
                "server_time_ms": (time.time() - server_start) * 1000
            }
        }
        
        # Update statistics
        self.stats["requests"] += 1
        self.stats["total_prediction_time"] += model_time
        self.stats["avg_prediction_time"] = self.stats["total_prediction_time"] / self.stats["requests"]
        self.stats["uptime_seconds"] = time.time() - self.stats["start_time"]
        
        return results
    
    def _get_batch_predictions(self, texts, top_k=3):
        """Get batch predictions for multiple texts"""
        server_start = time.time()
        batch_results = []
        
        # Use the batch_predict method from predictor if available
        if hasattr(self.predictor, 'batch_predict'):
            # Use the batch prediction functionality
            batch_predictions = self.predictor.batch_predict(texts, top_k=top_k)
            
            # Format each result
            for i, (predictions, model_time) in enumerate(batch_predictions):
                result = {
                    "predictions": [{"word": word, "probability": prob} for word, prob in predictions],
                    "metadata": {
                        "model_time_ms": model_time * 1000,
                        "server_time_ms": 0  # Will update at the end
                    }
                }
                batch_results.append(result)
                
                # Update statistics for each prediction
                self.stats["requests"] += 1
                self.stats["total_prediction_time"] += model_time
        else:
            # Fallback to individual predictions if batch_predict not available
            for text in texts:
                predictions, model_time = self.predictor.predict_next(text, top_k=top_k)
                
                result = {
                    "predictions": [{"word": word, "probability": prob} for word, prob in predictions],
                    "metadata": {
                        "model_time_ms": model_time * 1000,
                        "server_time_ms": 0  # Will update at the end
                    }
                }
                batch_results.append(result)
                
                # Update statistics for each prediction
                self.stats["requests"] += 1
                self.stats["total_prediction_time"] += model_time
        
        # Update average prediction time
        self.stats["avg_prediction_time"] = self.stats["total_prediction_time"] / self.stats["requests"]
        self.stats["uptime_seconds"] = time.time() - self.stats["start_time"]
        
        # Add total server processing time to metadata
        server_time = (time.time() - server_start) * 1000
        for result in batch_results:
            result["metadata"]["server_time_ms"] = server_time
        
        return batch_results

def main():
    """Run the LLM prediction server"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Lightweight LLM Prediction Server")
    parser.add_argument("--host", default="localhost", help="Host address (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    parser.add_argument("--disable-optimizations", action="store_true", 
                        help="Disable model optimizations for higher quality predictions")
    args = parser.parse_args()
    
    try:
        # Check if required packages are installed
        required_packages = ["torch", "transformers"]
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"Missing required packages: {', '.join(missing_packages)}")
            logger.error("Please install them with pip:")
            logger.error(f"pip install {' '.join(missing_packages)}")
            sys.exit(1)
        
        # Create and start the server
        server = LLMPredictionServer(
            host=args.host, 
            port=args.port,
            model_optimizations=not args.disable_optimizations
        )
        server.start()
        
    except Exception as e:
        logger.error(f"Error running server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()