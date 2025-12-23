"""
Parallel Processing Module for Football Prediction Hub
Phase 2.1 - Performance & Scalability Implementation
"""

import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Dict, Any, Tuple
import logging
import time
from datetime import datetime
import json
from functools import partial
import queue
import threading

logger = logging.getLogger(__name__)

class ParallelProcessor:
    """Handles parallel processing of predictions and API requests"""
    
    def __init__(self, max_workers=4, max_api_concurrent=10):
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=max_workers)
        self.max_api_concurrent = max_api_concurrent
        self.prediction_queue = queue.Queue()
        self.results_cache = {}
        self._start_worker_threads()
        
    def _start_worker_threads(self):
        """Start background worker threads for processing queue"""
        for i in range(2):  # 2 worker threads
            worker = threading.Thread(target=self._queue_worker, daemon=True)
            worker.start()
            
    def _queue_worker(self):
        """Worker thread that processes prediction queue"""
        while True:
            try:
                task = self.prediction_queue.get(timeout=1)
                if task is None:
                    break
                    
                task_id, func, args, kwargs = task
                try:
                    result = func(*args, **kwargs)
                    self.results_cache[task_id] = {'status': 'completed', 'result': result}
                except Exception as e:
                    logger.error(f"Task {task_id} failed: {str(e)}")
                    self.results_cache[task_id] = {'status': 'failed', 'error': str(e)}
                finally:
                    self.prediction_queue.task_done()
            except queue.Empty:
                continue
                
    async def fetch_multiple_apis_async(self, urls: List[str], headers: Dict = None) -> List[Dict]:
        """Fetch data from multiple APIs concurrently"""
        async with aiohttp.ClientSession() as session:
            tasks = []
            semaphore = asyncio.Semaphore(self.max_api_concurrent)
            
            async def fetch_with_semaphore(url):
                async with semaphore:
                    return await self._fetch_single_api(session, url, headers)
            
            for url in urls:
                task = fetch_with_semaphore(url)
                tasks.append(task)
                
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results
            
    async def _fetch_single_api(self, session: aiohttp.ClientSession, url: str, headers: Dict = None) -> Dict:
        """Fetch data from a single API endpoint"""
        try:
            start_time = time.time()
            async with session.get(url, headers=headers, timeout=30) as response:
                data = await response.json()
                elapsed = time.time() - start_time
                logger.info(f"API call to {url} completed in {elapsed:.2f}s")
                return {
                    'url': url,
                    'status': response.status,
                    'data': data,
                    'elapsed_time': elapsed
                }
        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching {url}")
            return {'url': url, 'error': 'timeout', 'status': 408}
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return {'url': url, 'error': str(e), 'status': 500}
            
    def batch_predict_parallel(self, match_pairs: List[Tuple[str, str, str, str]], predictor) -> List[Dict]:
        """Process multiple predictions in parallel"""
        start_time = time.time()
        
        # Create partial function with predictor
        predict_func = partial(self._single_prediction, predictor=predictor)
        
        # Execute predictions in parallel
        with self.thread_pool as executor:
            futures = []
            for home_id, away_id, home_name, away_name in match_pairs:
                future = executor.submit(predict_func, home_id, away_id, home_name, away_name)
                futures.append(future)
                
            # Collect results
            results = []
            for i, future in enumerate(futures):
                try:
                    result = future.result(timeout=60)  # 60 second timeout per prediction
                    results.append(result)
                except Exception as e:
                    logger.error(f"Prediction {i} failed: {str(e)}")
                    results.append({
                        'error': str(e),
                        'match': f"{match_pairs[i][2]} vs {match_pairs[i][3]}"
                    })
                    
        elapsed = time.time() - start_time
        logger.info(f"Batch prediction of {len(match_pairs)} matches completed in {elapsed:.2f}s")
        
        return results
        
    def _single_prediction(self, home_id: str, away_id: str, home_name: str, away_name: str, predictor) -> Dict:
        """Execute a single prediction"""
        try:
            return predictor.predict_match(home_id, away_id, home_name, away_name, force_update=True)
        except Exception as e:
            logger.error(f"Error predicting {home_name} vs {away_name}: {str(e)}")
            raise
            
    def queue_prediction(self, task_id: str, func, *args, **kwargs) -> str:
        """Queue a prediction task for background processing"""
        task = (task_id, func, args, kwargs)
        self.prediction_queue.put(task)
        self.results_cache[task_id] = {'status': 'queued'}
        return task_id
        
    def get_task_status(self, task_id: str) -> Dict:
        """Get the status of a queued task"""
        return self.results_cache.get(task_id, {'status': 'not_found'})
        
    def parallel_model_training(self, models: List[Any], training_data: Dict) -> List[Any]:
        """Train multiple models in parallel"""
        start_time = time.time()
        
        with self.process_pool as executor:
            futures = []
            for model in models:
                future = executor.submit(self._train_single_model, model, training_data)
                futures.append(future)
                
            trained_models = []
            for future in futures:
                try:
                    trained_model = future.result(timeout=300)  # 5 minute timeout
                    trained_models.append(trained_model)
                except Exception as e:
                    logger.error(f"Model training failed: {str(e)}")
                    trained_models.append(None)
                    
        elapsed = time.time() - start_time
        logger.info(f"Parallel training of {len(models)} models completed in {elapsed:.2f}s")
        
        return trained_models
        
    def _train_single_model(self, model: Any, training_data: Dict) -> Any:
        """Train a single model (to be run in separate process)"""
        try:
            # Model-specific training logic would go here
            # This is a placeholder for the actual training
            logger.info(f"Training model {type(model).__name__}")
            # model.fit(training_data)
            return model
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
            
    def cleanup(self):
        """Cleanup resources"""
        # Signal workers to stop
        for _ in range(2):
            self.prediction_queue.put(None)
        
        # Shutdown executors
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        
        
class BatchPredictionManager:
    """Manages batch prediction operations with progress tracking"""
    
    def __init__(self, parallel_processor: ParallelProcessor):
        self.processor = parallel_processor
        self.batch_status = {}
        
    def create_batch(self, batch_id: str, matches: List[Dict]) -> Dict:
        """Create a new batch prediction job"""
        self.batch_status[batch_id] = {
            'id': batch_id,
            'total': len(matches),
            'completed': 0,
            'failed': 0,
            'status': 'processing',
            'created_at': datetime.now().isoformat(),
            'matches': matches,
            'results': []
        }
        return self.batch_status[batch_id]
        
    def update_batch_progress(self, batch_id: str, completed: int = 0, failed: int = 0):
        """Update batch processing progress"""
        if batch_id in self.batch_status:
            self.batch_status[batch_id]['completed'] += completed
            self.batch_status[batch_id]['failed'] += failed
            
            total = self.batch_status[batch_id]['total']
            done = self.batch_status[batch_id]['completed'] + self.batch_status[batch_id]['failed']
            
            if done >= total:
                self.batch_status[batch_id]['status'] = 'completed'
                self.batch_status[batch_id]['completed_at'] = datetime.now().isoformat()
                
    def get_batch_status(self, batch_id: str) -> Dict:
        """Get current status of a batch"""
        return self.batch_status.get(batch_id, {'status': 'not_found'})
        
    def process_batch_async(self, batch_id: str, predictor):
        """Process a batch asynchronously"""
        if batch_id not in self.batch_status:
            return {'error': 'Batch not found'}
            
        batch = self.batch_status[batch_id]
        matches = batch['matches']
        
        # Queue all predictions
        for match in matches:
            task_id = f"{batch_id}_{match['home_id']}_{match['away_id']}"
            self.processor.queue_prediction(
                task_id,
                predictor.predict_match,
                match['home_id'],
                match['away_id'],
                match['home_name'],
                match['away_name'],
                force_update=True
            )
            
        return {'status': 'queued', 'batch_id': batch_id}