#!/usr/bin/env python3
import os
import logging
import time
import threading
from typing import Dict, Any, List, Optional

# LLM API clients
from openai import OpenAI
import anthropic
import google.generativeai as genai
try:
    from deepseek import DeepSeekAPI
except ImportError:
    # Placeholder for DeepSeek API if not available
    class DeepSeekAPI:
        def __init__(self, api_key):
            self.api_key = api_key

# Get logger
logger = logging.getLogger(__name__)

class ResponseAnalyzer:
    """
    Analyzes model responses using various LLM APIs
    """
    
    def __init__(self, llm_type: str = "openai", temperature: float = 0.3, max_tokens: int = 1000, max_retries: int = 50):
        """
        Initialize the response analyzer
        
        Args:
            llm_type: Type of LLM API to use ('openai', 'claude', 'gemini', or 'deepseek')
            temperature: Temperature for LLM generation
            max_tokens: Maximum tokens for LLM response
            max_retries: Maximum number of retry attempts when API calls fail
        """
        self.llm_type = llm_type.lower()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.client = self._initialize_client()
        # Add a lock to control API call rate across threads
        self.api_lock = threading.Lock()
        self.last_call_time = 0
        self.rate_limit_delay = 0.5  # Default delay between API calls (in seconds)
        
    def _initialize_client(self):
        """Initialize the appropriate LLM client based on llm_type"""
        if self.llm_type == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            return OpenAI(api_key=api_key)
            
        elif self.llm_type == "claude":
            # api_key = os.getenv("ANTHROPIC_API_KEY")
            from environment import ANTHROPIC_API_KEY
            api_key = ANTHROPIC_API_KEY
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
            return anthropic.Anthropic(api_key=api_key)
            
        elif self.llm_type == "gemini":
            # api_key = os.getenv("GOOGLE_API_KEY")
            #key is in environment.py
            from environment import GEMINI_API_KEY
            api_key = GEMINI_API_KEY
            
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable is not set")
            genai.configure(api_key=api_key)
            return genai
            
        elif self.llm_type == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY environment variable is not set")
            return DeepSeekAPI(api_key=api_key)
            
        else:
            raise ValueError(f"Unsupported LLM type: {self.llm_type}")
    
    def set_rate_limit(self, delay_seconds: float) -> None:
        """
        Set the rate limit delay between API calls
        
        Args:
            delay_seconds: Delay in seconds between API calls
        """
        self.rate_limit_delay = max(0, delay_seconds)
        logger.info(f"Rate limit for {self.llm_type} API set to {self.rate_limit_delay} seconds")
    
    def _enforce_rate_limit(self) -> None:
        """
        Enforce rate limiting between API calls
        This method is thread-safe and will ensure proper spacing between API calls
        """
        with self.api_lock:
            current_time = time.time()
            time_since_last_call = current_time - self.last_call_time
            
            if time_since_last_call < self.rate_limit_delay:
                sleep_time = self.rate_limit_delay - time_since_last_call
                logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
            
            self.last_call_time = time.time()
    
    def analyze_response(self, prompt: str) -> str:
        """
        Analyze using the selected LLM API with retry logic
        
        Args:
            prompt: The full prompt to send to the LLM
            
        Returns:
            LLM output as string
        
        Raises:
            Exception: If all retry attempts fail
        """
        retry_count = 0
        last_error = None
        
        while retry_count < self.max_retries:
            # Apply rate limiting before making the API call
            self._enforce_rate_limit()
            
            try:
                if self.llm_type == "openai":
                    return self._analyze_with_openai(prompt)
                elif self.llm_type == "claude":
                    return self._analyze_with_claude(prompt)
                elif self.llm_type == "gemini":
                    return self._analyze_with_gemini(prompt)
                elif self.llm_type == "deepseek":
                    return self._analyze_with_deepseek(prompt)
            except Exception as e:
                retry_count += 1
                last_error = e
                
                # Different wait times based on retry count (exponential backoff)
                wait_time = min(2 ** retry_count, 60)  # Cap at 60 seconds wait
                
                logger.warning(
                    f"Error calling {self.llm_type} API: {str(e)}. "
                    f"Retry {retry_count}/{self.max_retries} after {wait_time} seconds."
                )
                
                # Wait before retrying
                time.sleep(wait_time)
        
        # If we've exhausted all retries, log the error and re-raise
        logger.error(f"Failed after {self.max_retries} attempts. Last error: {str(last_error)}")
        raise last_error or Exception(f"Failed to get response after {self.max_retries} attempts")
    
    def _analyze_with_openai(self, prompt: str) -> str:
        """Use OpenAI API with full prompt"""
        completion = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts information from model outputs."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return completion.choices[0].message.content
    
    def _analyze_with_claude(self, prompt: str) -> str:
        """Use Anthropic's Claude API with full prompt"""
        message = self.client.messages.create(
            model="claude-3-7-sonnet-20250219",
            system="You are a helpful assistant that extracts information from model outputs.",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return message.content[0].text
    
    def _analyze_with_gemini(self, prompt: str) -> str:
        """Use Google's Gemini API with full prompt"""
        model = self.client.GenerativeModel('gemini-2.5-pro-preview-03-25')
        # model = self.client.GenerativeModel('gemini-2.0-flash')
        result = model.generate_content(
            f"System: You are a helpful assistant that extracts information from model outputs.\n\nUser: {prompt}"
        )
        return result.text
    
    def _analyze_with_deepseek(self, prompt: str) -> str:
        """Use DeepSeek API with full prompt"""
        result = self.client.chat(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts information from model outputs."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return result.choices[0].message.content


def get_available_llm_types() -> List[str]:
    """Get the list of available LLM types"""
    return ["openai", "claude", "gemini", "deepseek"]


# API testing functions
def test_openai_client() -> bool:
    """Test if the OpenAI client can be initialized"""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY environment variable is not set.")
            return False
            
        client = OpenAI(api_key=api_key)
        logger.info("OpenAI client initialized successfully")
        return True
    except Exception as e:
        logger.error(f"ERROR initializing OpenAI client: {e}")
        return False

def test_claude_client() -> bool:
    """Test if the Claude client can be initialized"""
    try:
        # api_key = os.getenv("ANTHROPIC_API_KEY")
        from environment import ANTHROPIC_API_KEY
        api_key = ANTHROPIC_API_KEY
        # import pdb; pdb.set_trace()
        if not api_key:
            logger.error("ANTHROPIC_API_KEY environment variable is not set.")
            return False
            
        client = anthropic.Anthropic(api_key=api_key)
        logger.info("Claude client initialized successfully")
        return True
    except Exception as e:
        logger.error(f"ERROR initializing Claude client: {e}")
        return False

def test_gemini_client() -> bool:
    """Test if the Gemini client can be initialized"""
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error("GOOGLE_API_KEY environment variable is not set.")
            return False
            
        genai.configure(api_key=api_key)
        logger.info("Gemini client initialized successfully")
        return True
    except Exception as e:
        logger.error(f"ERROR initializing Gemini client: {e}")
        return False

def test_deepseek_client() -> bool:
    """Test if the DeepSeek client can be initialized"""
    try:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            logger.error("DEEPSEEK_API_KEY environment variable is not set.")
            return False
            
        client = DeepSeekAPI(api_key=api_key)
        logger.info("DeepSeek client initialized successfully")
        return True
    except Exception as e:
        logger.error(f"ERROR initializing DeepSeek client: {e}")
        return False

# Dictionary of test functions for each LLM type
test_functions = {
    'openai': test_openai_client,
    'claude': test_claude_client,
    'gemini': test_gemini_client,
    'deepseek': test_deepseek_client
} 