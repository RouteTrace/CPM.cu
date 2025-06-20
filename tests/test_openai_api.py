#!/usr/bin/env python3
import requests
import json
import sys
import argparse

def test_chat_completion(stream=True):
    """Test chat completion with streaming or non-streaming mode"""
    url = "http://localhost:8000/v1/chat/completions"
    
    data = {
        "model": "model",
        "messages": [
            {"role": "user", "content": "Please tell me how to write quick sort algorithm"}
        ],
        "stream": stream,
        "max_tokens": 1024,
        "temperature": 0.0
    }
    
    mode = "streaming" if stream else "non-streaming"
    print(f"Testing {mode} chat completion...")
    print("Request:", json.dumps(data, indent=2, ensure_ascii=False))
    print(f"\n{mode.capitalize()} response:")
    print("-" * 50)
    
    try:
        response = requests.post(url, json=data, stream=stream, timeout=60)
        response.raise_for_status()
        
        if stream:
            return _handle_streaming_response(response)
        else:
            return _handle_non_streaming_response(response)
            
    except requests.exceptions.Timeout:
        print(f"\nRequest timed out")
        return False
    except requests.exceptions.RequestException as e:
        print(f"\nRequest failed: {e}")
        return False

def _handle_streaming_response(response):
    """Handle streaming response"""
    print("Connected, waiting for response...")
    sys.stdout.flush()
    
    for line in response.iter_lines(decode_unicode=True):
        if line and line.startswith('data: '):
            data_str = line[6:]  # Remove 'data: ' prefix
            
            # Check for end of stream
            if data_str.strip() == '[DONE]':
                print("\n[DONE]")
                break
            
            # Skip empty data
            if not data_str.strip():
                continue
                
            try:
                chunk = json.loads(data_str)
                
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    choice = chunk['choices'][0]
                    
                    # Handle content delta
                    if 'delta' in choice and 'content' in choice['delta']:
                        content = choice['delta']['content']
                        if content:  # Only print non-empty content
                            print(content, end='', flush=True)
                    
                    # Handle finish reason
                    elif choice.get('finish_reason'):
                        print(f"\n[Finished: {choice['finish_reason']}]")
                        break
                        
            except json.JSONDecodeError as e:
                print(f"\nError parsing JSON: {e}")
                print(f"Raw data: {repr(data_str)}")
                continue
    
    return True

def _handle_non_streaming_response(response):
    """Handle non-streaming response"""
    try:
        result = response.json()
        print("Response received!")
        print("Full response:", json.dumps(result, indent=2, ensure_ascii=False))
        
        # Extract and display the content
        if 'choices' in result and len(result['choices']) > 0:
            choice = result['choices'][0]
            if 'message' in choice and 'content' in choice['message']:
                content = choice['message']['content']
                print("\nGenerated content:")
                print("-" * 30)
                print(content)
                print("-" * 30)
                
                # Display usage statistics if available
                if 'usage' in result:
                    usage = result['usage']
                    print(f"\nTokens used: {usage.get('total_tokens', 'N/A')}")
                    print(f"Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
                    print(f"Completion tokens: {usage.get('completion_tokens', 'N/A')}")
            else:
                print("No message content found in response")
        else:
            print("No choices found in response")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"\nError parsing response JSON: {e}")
        return False

def check_server_health():
    """Check if server is running and model is loaded"""
    try:
        health_response = requests.get("http://localhost:8000/health")
        health_response.raise_for_status()
        health_data = health_response.json()
        print("Health check:", health_data)
        
        if not health_data.get('model_loaded', False):
            print("Model not loaded! Please start the server first.")
            return False
            
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Health check failed: {e}")
        print("Please make sure the server is running on localhost:8000")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPM.cu Server API Test")
    parser.add_argument('--no-stream', action='store_true', 
                       help='Use non-streaming mode instead of streaming (default: streaming)')
    args = parser.parse_args()
    
    use_stream = not args.no_stream
    mode = "Streaming" if use_stream else "Non-streaming"
    print(f"CPM.cu Server {mode} Test")
    print("=" * 50)
    
    # Check server health first
    if not check_server_health():
        sys.exit(1)
    
    # Run test
    success = test_chat_completion(stream=use_stream)
    
    if success:
        print(f"\n{mode} test completed successfully!")
    else:
        print(f"\n{mode} test failed!")
        sys.exit(1)
